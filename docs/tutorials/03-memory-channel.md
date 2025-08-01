# Memory Channel

```{note}
This tutorial follows the [Bootstrap and Communicator](./02-bootstrap-and-communicator.md) tutorial.
```

In this tutorial, we will introduce comprehensive usage of `MemoryChannel`, which provides direct access to remote GPU memory for communication. We will cover how to create communication buffers, how to use them with `MemoryChannel`, and how to perform efficient data transfer between GPUs using `MemoryChannel`.

## Build and Run the Example

The code of this tutorial is under [examples/tutorials/03-memory-channel](https://github.com/microsoft/mscclpp/blob/main/examples/tutorials/03-memory-channel).

Build the example with `make`:

```bash
$ cd examples/tutorials/03-memory-channel
$ make
```

Run the example with `./perf_memory_channel`. If you are in a container, you may need to run with root privileges. You should see output similar to the following:

```
# ./perf_memory_channel
```
