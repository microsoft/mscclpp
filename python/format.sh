#!/bin/bash

clang-format \
	-style='{"BasedOnStyle": "google", "BinPackParameters": false, "BinPackArguments": false, "AlignAfterOpenBracket": "AlwaysBreak"}' \
	-i src/*

