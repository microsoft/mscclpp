# Python bindings

Test instructions:
 * Compile the `libmscclpp.so` library.
 * Install `cmake` verion >= 3.18
 * setup a python virtual env
   * `pip install -r requirements.txt`
 * `./tesh.sh`

Rough build attemtps
```
# cd to this directory:

# setup/enter pyenv environment for python 3.9

# install nanabind and the test requirements.
pip install -r requirements.txt

# setup and build the CMake environments.
# this requires nanobind, installed above.
./setup.sh

# test the module
pytest build/mscclpp
```


## Installing `gdrcopy` and `mpi`
This assumes that some things are built/installed
```
# assumes WORKDIR has:
# git clone git@github.com/NVIDIA/gdrcopy.git
# git clone git@github.com:microsoft/mscclpp.git

uname -r
# 5.4.0-1090-azure

# install

# break /usr/sbin/policy-rc.d so we can install modules
echo '#!/bin/sh
exit 0' > /usr/sbin/policy-rc.d

apt update
apt install -y \
  build-essential devscripts debhelper check \
  libsubunit-dev fakeroot pkg-config dkms \
  linux-headers-5.4.0-1090-azure 
  
apt install -y nvidia-dkms-525-server


cd $WORKDIR/gdrcopy
sed -i 's/\(-L \$(CUDA)\/lib64\)/\1 \1\/stubs/' tests/Makefile
cd packages
CUDA=/usr/local/cuda ./build-deb-packages.sh

dpkg -i gdrdrv-dkms_2.3-1_amd64.Ubuntu20_04.deb
dpkg -i libgdrapi_2.3-1_amd64.Ubuntu20_04.deb
dpkg -i gdrcopy-tests_2.3-1_amd64.Ubuntu20_04+cuda11.6.deb
dpkg -i gdrcopy_2.3-1_amd64.Ubuntu20_04.deb

# validate:
# $ sanity
# Running suite(s): Sanity
# 100%: Checks: 27, Failures: 0, Errors: 0

# dkms install -m gdrdrv/2.3

cd $WORKDIR/mscclpp

## numctl
apt install -y numactl libnuma-dev libnuma1

# if not mpi testing
USE_MPI_FOR_TESTS=0 make -j
```
