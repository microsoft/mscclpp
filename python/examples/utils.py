# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import time

import mscclpp


def main():
    timer = mscclpp.Timer()
    timer.reset()
    time.sleep(2)
    assert timer.elapsed() >= 2000000


if __name__ == "__main__":
    main()
