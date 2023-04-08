import concurrent.futures
import os
import subprocess
import sys
import unittest

import hamcrest

import mscclpp

MOD_DIR = os.path.dirname(__file__)
TESTS_DIR = os.path.join(MOD_DIR, "tests")


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
            hamcrest.calling(mscclpp.MscclppUniqueId.from_bytes).with_args(b"abc"),
            hamcrest.raises(
                ValueError,
                f"Requires exactly {mscclpp.MSCCLPP_UNIQUE_ID_BYTES} bytes; found 3",
            ),
        )


class CommsTest(unittest.TestCase):
    def test_all_gather(self) -> None:
        world_size = 2

        tasks: list[concurrent.futures.Future[None]] = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=world_size) as pool:
            for rank in range(world_size):
                tasks.append(
                    pool.submit(
                        subprocess.check_output,
                        [
                            "python",
                            "-m",
                            "mscclpp.tests.bootstrap_test",
                            f"--rank={rank}",
                            f"--world_size={world_size}",
                        ],
                        stderr=subprocess.STDOUT,
                    )
                )

        errors = []
        for rank, f in enumerate(tasks):
            try:
                f.result()
            except subprocess.CalledProcessError as e:
                errors.append((rank, e.output))

        if errors:
            parts = []
            for rank, content in errors:
                parts.append(
                    f"[rank {rank}]: " + content.decode("utf-8", errors="ignore")
                )

            raise AssertionError("\n\n".join(parts))
