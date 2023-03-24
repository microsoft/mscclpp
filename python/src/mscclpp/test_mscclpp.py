import unittest
import hamcrest

import mscclpp

class DTypeTest(unittest.TestCase):
    def test(self) -> None:
        for name, val in [
            ('int8', 0),
            ('char', 0),
            ('uint8', 1),
            ('int32', 2),
            ('int', 2),
            ('uint32', 3),
            ('int64', 4),
            ('uint64', 5),
            ('float16', 6),
            ('half', 6),
            ('float32', 7),
            ('float', 7),
            ('float64', 8),
            ('double', 8),
        ]:
            try:
                dtype = getattr(mscclpp.dtype, name)
                hamcrest.assert_that(
                    mscclpp.dtype(val),
                    hamcrest.equal_to(dtype),
                    reason=(name, val),
                )
                hamcrest.assert_that(
                    int(mscclpp.dtype(val)),
                    hamcrest.equal_to(val),
                    reason=(name, val),
                )
            except Exception as e:
                raise AssertionError((name, val)) from e

class ReduceOpTest(unittest.TestCase):
    def test(self) -> None:
        for name, val in [
            ('sum', 0),
            ('prod', 1),
            ('max', 2),
            ('min', 3),
            ('avg', 4),
        ]:
            try:
                dtype = getattr(mscclpp.reduce_op, name)
                hamcrest.assert_that(
                    mscclpp.reduce_op(val),
                    hamcrest.equal_to(dtype),
                    reason=(name, val),
                )
                hamcrest.assert_that(
                    int(mscclpp.reduce_op(val)),
                    hamcrest.equal_to(val),
                    reason=(name, val),
                )
            except Exception as e:
                raise AssertionError((name, val)) from e


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


class CommsTest(unittest.TestCase):
    def _test(self) -> None:
        # this hangs forever
        comm = mscclpp.MscclppComm.init_rank_from_address(
            address="127.0.0.1:50000",
            rank=0,
            world_size=2,
        )
        comm.close()
