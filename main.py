#!/usr/bin/env python3
"""Compatibility shim: `python main.py ...` == the `flowcount` console script.

The real CLI lives in flowcount/cli.py; install with `pip install -e .` and
run `flowcount --input 0 --live` from anywhere.
"""

from flowcount.cli import main

if __name__ == "__main__":
    main()
