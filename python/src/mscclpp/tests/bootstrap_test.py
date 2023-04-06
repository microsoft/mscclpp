import argparse
from dataclasses import dataclass

import hamcrest

import mscclpp


@dataclass
class Example:
    rank: int


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rank", type=int, required=True)
    p.add_argument("--world_size", type=int, required=True)
    p.add_argument("--port", default=50000)
    options = p.parse_args()

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

    hamcrest.assert_that(
        comm.bootstrap_all_gather_int(options.rank + 42),
        hamcrest.equal_to(
            [
                42,
                43,
            ]
        ),
    )

    hamcrest.assert_that(
        comm.all_gather_bytes(b"abc" * (1 + options.rank)),
        hamcrest.equal_to(
            [
                b"abc",
                b"abcabc",
            ]
        ),
    )

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

    hamcrest.assert_that(
        comm.all_gather_pickle(Example(rank=options.rank)),
        hamcrest.equal_to(
            [
                Example(rank=0),
                Example(rank=1),
            ]
        ),
    )

    comm.close()


if __name__ == "__main__":
    main()
