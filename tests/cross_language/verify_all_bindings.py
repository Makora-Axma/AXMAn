#!/usr/bin/env python3
"""
AXMANet Cross-Language Verification CLI Script (Phase 4.5).

Runs the golden sample through all 4 native wrappers (C++, Rust, Python, Java)
and verifies bit-exact structural hash and numerical parity with epsilon <= 1e-6.
"""

import sys
from pathlib import Path

# Add tests/cross_language to path
script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir))

from test_cross_language_parity import test_cross_language_numerical_identity

def main():
    print("=" * 65)
    print("      AXMANet PHASE 4.5: CROSS-LANGUAGE NUMERICAL VERIFICATION   ")
    print("=" * 65)
    try:
        test_cross_language_numerical_identity()
        return 0
    except Exception as e:
        print(f"\n[ERROR] Cross-language verification failed: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
