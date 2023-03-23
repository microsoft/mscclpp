This assumes that some things are built/installed
```
# assumes WORKDIR has:
# git clone git@github.com/NVIDIA/gdrcopy.git
# git clone git@github.com:microsoft/mscclpp.git

uname -r
# 5.4.0-1090-azure

# install

apt update
apt install -y \
  build-essential devscripts debhelper check \
  libsubunit-dev fakeroot pkg-config dkms \
  nvidia-dkms-525-server \
  linux-headers-5.4.0-1090-azure 


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


Rough build attemtps
```
# cd to this directory:

cmake -S . -B build
cmake --build build --clean-first -v

# this should contain libmscclpp.so, but does not
ldd build/py_mscclpp.cpython-39-x86_64-linux-gnu.so

# this will fail due to a missing symbol
( cd build;
  LD_LIBRARY_PATH="$PWD/../../build/lib:$LD_LIBRARY_PATH" python -c 'import py_mscclpp' )
```
