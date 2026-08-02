#!/usr/bin/env python3
"""Public entry point for the all-ansatz new-data integration workflow."""

try:
    from integrate_0730_twoc3 import main
except ImportError:
    from .integrate_0730_twoc3 import main


if __name__ == "__main__":
    raise SystemExit(main())
