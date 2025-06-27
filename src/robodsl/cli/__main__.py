#!/usr/bin/env python3
"""RoboDSL package entry point.

Running `python -m robodsl` should behave exactly like running the
stand-alone CLI defined in ``robodsl.cli``.  To avoid code duplication we
simply re-export and invoke that Click-based CLI here.
"""

from robodsl.cli import main as _cli_main

# When the package is executed via ``python -m robodsl`` this file is
# executed and we immediately delegate to the Click CLI defined in
# ``robodsl.cli``.  All command implementations live there so that the
# CLI has a single source of truth.
if __name__ == "__main__":
    _cli_main()
