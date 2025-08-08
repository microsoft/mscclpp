# Port Channel

```{note}
This tutorial follows the [Memory Channel](./03-memory-channel.md) tutorial.
```

## Build and Run the Example

The code of this tutorial is under [examples/tutorials/04-port-channel](https://github.com/microsoft/mscclpp/blob/main/examples/tutorials/04-port-channel).

Build the example with `make`:

```bash
$ cd examples/tutorials/04-port-channel
$ make
```

Run the example with `./bidir_port_channel`. If you are in a container, you may need to run with root privileges. You should see output similar to the following:

```
# ./bidir_port_channel
GPU 0: Preparing for tests ...
GPU 1: Preparing for tests ...
GPU 0: [Bidir PutWithSignal] bytes 1024, elapsed 0.0204875 ms/iter, BW 0.0499818 GB/s
GPU 0: [Bidir PutWithSignal] bytes 1048576, elapsed 0.0250319 ms/iter, BW 41.8896 GB/s
GPU 0: [Bidir PutWithSignal] bytes 134217728, elapsed 0.365497 ms/iter, BW 367.219 GB/s
Succeed!
```

The example code uses localhost port `50505` by default. If the port is already in use, you can change it by modifying the `PORT_NUMBER` macro in the code.

```{caution}
Note that this example is **NOT** a performance benchmark. The performance numbers are provided to give you an idea of the performance characteristics of `PortChannel`. For optimal performance, synchronization can be further optimized depending on the application scenario and implementation.
```

## Code Overview

TBU
