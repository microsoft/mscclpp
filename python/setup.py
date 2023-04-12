#!/usr/bin/env python

import os
import shutil
import sys
import logging
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext
import subprocess

THIS_DIR = os.path.abspath(os.path.dirname(__file__))


class CustomExt(Extension):
    def __init__(self, name):
        # don't invoke the original build_ext for this special extension
        super().__init__(name, sources=[])


class custom_build_ext(build_ext):
    def run(self):
        for ext in self.extensions:
            if isinstance(ext, CustomExt):
                self.build_extension(ext)
            else:
                super().run()

    def build_extension(self, ext):
        if sys.platform == "darwin":
            return

        # these dirs will be created in build_py, so if you don't have
        # any python sources to bundle, the dirs will be missing
        build_temp = os.path.abspath(self.build_temp)
        os.makedirs(build_temp, exist_ok=True)

        try:
            subprocess.check_output(
                ["cmake", "-S", THIS_DIR, "-B", build_temp],
                stderr=subprocess.STDOUT,
            )
            subprocess.check_output(
                ["cmake", "--build", build_temp],
                stderr=subprocess.STDOUT,
            )
        except subprocess.CalledProcessError as e:
            logging.error(e.output.decode())
            raise

        libname = os.path.basename(self.get_ext_fullpath(ext.name))

        target_dir = os.path.join(
            os.path.dirname(self.get_ext_fullpath(ext.name)),
            "mscclpp",
        )

        shutil.copy(
            os.path.join(build_temp, "libmscclpp.so"),
            target_dir,
        )

        shutil.copy(
            os.path.join(build_temp, libname),
            target_dir,
        )


setup(
    name='mscclpp',
    version='0.1.0',
    description='Python bindings for mscclpp',
    # packages=['mscclpp'],
    package_dir={'': 'src'},
    packages=find_packages(where='./src'),
    ext_modules=[CustomExt('_py_mscclpp')],
    cmdclass={
        "build_ext": custom_build_ext,
    },
    install_requires=[
        'torch',
        'nanobind',
    ],
)
