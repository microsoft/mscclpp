# Python bindings

Test instructions:

* Compile the `libmscclpp.so` library.
* Install `cmake` verion >= 3.18
* setup a python virtual env
    * `pip install -r dev-requirements.txt`
* `./tesh.sh`

## Run CI:

```bash
./ci.sh
```

## Build a wheel:

Setup dev environment, then:

```bash
python setup.py bdist_wheel
```

## Installing mpi and numa libs.

```
## numctl
apt install -y numactl libnuma-dev libnuma1
```
