import argparse
import os
from dataclasses import dataclass

import hamcrest
import torch

import mscclpp


@dataclass
class Example:
    rank: int


def _test_allgather_int(options: argparse.Namespace, comm: mscclpp.Comm):
    hamcrest.assert_that(
        comm.bootstrap_all_gather_int(options.rank + 42),
        hamcrest.equal_to(
            [
                42,
                43,
            ]
        ),
    )


def _test_allgather_bytes(options: argparse.Namespace, comm: mscclpp.Comm):
    hamcrest.assert_that(
        comm.all_gather_bytes(b"abc" * (1 + options.rank)),
        hamcrest.equal_to(
            [
                b"abc",
                b"abcabc",
            ]
        ),
    )


def _test_allgather_json(options: argparse.Namespace, comm: mscclpp.Comm):
    hamcrest.assert_that(
        comm.all_gather_json({"rank": options.rank}),
        hamcrest.equal_to(
            [
                {"rank": 0},
                {"rank": 1},
            ]
        ),
    )

    hamcrest.assert_that(
        comm.all_gather_json([options.rank, 42]),
        hamcrest.equal_to(
            [
                [0, 42],
                [1, 42],
            ]
        ),
    )


def _test_allgather_pickle(options: argparse.Namespace, comm: mscclpp.Comm):
    hamcrest.assert_that(
        comm.all_gather_pickle(Example(rank=options.rank)),
        hamcrest.equal_to(
            [
                Example(rank=0),
                Example(rank=1),
            ]
        ),
    )

    comm.connection_setup()


def _test_allgather_torch(options: argparse.Namespace, comm: mscclpp.Comm):
    buf = torch.zeros(
        [options.world_size], dtype=torch.int64, device="cuda"
    ).contiguous()
    rank = options.rank
    tag = 0
    remote_rank = (options.rank + 1) % options.world_size
    comm.connect(
        remote_rank,
        tag,
        buf.data_ptr(),
        buf.element_size() * buf.numel(),
        mscclpp._py_mscclpp.TransportType.P2P,
    )

    comm.connection_setup()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rank", type=int, required=True)
    p.add_argument("--world_size", type=int, required=True)
    p.add_argument("--port", default=50000)
    options = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(options.rank)

    comm_options = dict(
        address=f"127.0.0.1:{options.port}",
        rank=options.rank,
        world_size=options.world_size,
    )
    print(f"{comm_options=}", flush=True)

    comm = mscclpp.Comm.init_rank_from_address(**comm_options)
    # comm.connection_setup()

    hamcrest.assert_that(comm.rank, hamcrest.equal_to(options.rank))
    hamcrest.assert_that(comm.world_size, hamcrest.equal_to(options.world_size))

    try:
        _test_allgather_int(options, comm)
        _test_allgather_bytes(options, comm)
        _test_allgather_json(options, comm)
        _test_allgather_pickle(options, comm)
        _test_allgather_torch(options, comm)
    finally:
        comm.close()


if __name__ == "__main__":
    main()
