#!/usr/bin/env python3
"""
Top-level entrypoint for AXMANet cross-language verification.
"""

import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent
cross_lang_dir = root_dir / "tests" / "cross_language"
sys.path.insert(0, str(cross_lang_dir))

import verify_all_bindings

if __name__ == "__main__":
    sys.exit(verify_all_bindings.main())
