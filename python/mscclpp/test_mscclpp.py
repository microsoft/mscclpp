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


class CommsTest(unittest.TestCase):
    def _test(self) -> None:
        # this hangs forever
        comm = mscclpp.MscclppComm.init_rank_from_address(
            address="127.0.0.1:50000",
            rank=0,
            world_size=2,
        )
        comm.close()
