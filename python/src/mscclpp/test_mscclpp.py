import concurrent.futures
import unittest
import hamcrest

import mscclpp


class UniqueIdTest(unittest.TestCase):
    def test_no_constructor(self) -> None:
        hamcrest.assert_that(
            hamcrest.calling(mscclpp.MscclppUniqueId).with_args(),
            hamcrest.raises(
                TypeError,
                "no constructor",
            ),
        )

    def test_getUniqueId(self) -> None:
        myId = mscclpp.MscclppUniqueId.from_context()

        hamcrest.assert_that(
            myId.bytes(),
            hamcrest.has_length(mscclpp.MSCCLPP_UNIQUE_ID_BYTES),
        )

        # from_bytes should work
        copy = mscclpp.MscclppUniqueId.from_bytes(myId.bytes())
        hamcrest.assert_that(
            copy.bytes(),
            hamcrest.equal_to(myId.bytes()),
        )

        # bad size
        hamcrest.assert_that(
            hamcrest.calling(mscclpp.MscclppUniqueId.from_bytes).with_args(b'abc'),
            hamcrest.raises(
                ValueError,
                f"Requires exactly {mscclpp.MSCCLPP_UNIQUE_ID_BYTES} bytes; found 3"
            ),
        )

def all_gather_task(rank: int, world_size: int) -> None:
    comm_options = dict(
        address="127.0.0.1:50000",
        rank=rank,
        world_size=world_size,
    )
    print(f'{comm_options=}', flush=True)

    comm = mscclpp.MscclppComm.init_rank_from_address(**comm_options)

    buf = bytearray(world_size)
    buf[rank] = rank

    if False:
        # crashes, bad call structure..
        comm.bootstrap_all_gather(memoryview(buf), world_size)
        hamcrest.assert_that(
            buf,
            hamcrest.equal_to(b'\000\002'),
        )

    comm.close()


class CommsTest(unittest.TestCase):
    def test_all_gather(self) -> None:
        world_size = 2

        tasks: list[concurrent.futures.Future[None]] = []

        with concurrent.futures.ProcessPoolExecutor(max_workers=world_size) as pool:
            for rank in range(world_size):
                tasks.append(pool.submit(all_gather_task, rank, world_size))

        for f in concurrent.futures.as_completed(tasks):
            f.result()


