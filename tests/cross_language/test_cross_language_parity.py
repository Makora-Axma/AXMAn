#!/usr/bin/env python3
"""
Cross-Language Numerical Parity & Bit-Level Verification Test.

Runs and verifies output tensors across all 4 wrappers:
1. C++20 (axman-cpp)
2. Rust (axman-rs)
3. Python (axman-py)
4. Java 22+ (axman-java Panama FFM)

Verifies:
- Structural hash exact match (bit-exact uint64).
- Value estimate tolerance eps <= 1e-6.
- Policy logits max absolute difference eps <= 1e-6.
- Latent embedding max absolute difference eps <= 1e-6.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
import numpy as np

def run_cmd(cmd, cwd=None, env=None):
    print(f"Running: {cmd}")
    res = subprocess.run(cmd, shell=True, cwd=cwd, env=env, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"STDOUT: {res.stdout}")
        print(f"STDERR: {res.stderr}")
        raise RuntimeError(f"Command failed with code {res.returncode}: {cmd}")
    return res.stdout.strip()

def test_cross_language_numerical_identity():
    axman_root = Path(__file__).resolve().parent.parent.parent
    test_dir = axman_root / "tests" / "cross_language"
    
    # Step 1: Ensure dataset generated
    gen_script = test_dir / "generate_golden.py"
    run_cmd(f'"{sys.executable}" "{gen_script}"', cwd=str(axman_root))

    # Step 2: Execute C++ runner
    cpp_exe = axman_root / "build" / "runner_cpp.exe"
    run_cmd(f'"{cpp_exe}" "{test_dir}"', cwd=str(axman_root))

    # Step 3: Execute Rust runner
    env = os.environ.copy()
    env["PATH"] = str(axman_root) + os.pathsep + env.get("PATH", "")
    rust_exe = axman_root / "bindings" / "rust" / "target" / "debug" / "examples" / "runner_rust.exe"
    run_cmd(f'"{rust_exe}" "{test_dir}"', cwd=str(axman_root), env=env)

    # Step 4: Execute Python runner
    py_script = test_dir / "runner_python.py"
    run_cmd(f'"{sys.executable}" "{py_script}" "{test_dir}"', cwd=str(axman_root))

    # Step 5: Execute Java runner
    java_exe = "C:\\Program Files\\Java\\jdk-22\\bin\\java.exe"
    cp = str(axman_root / "bindings" / "java" / "target" / "classes")
    run_cmd(f'"{java_exe}" --enable-preview --enable-native-access=ALL-UNNAMED -cp "{cp}" org.axma.axman.RunnerJava "{test_dir}"', cwd=str(axman_root))

    # Step 6: Load outputs and compare
    languages = ["cpp", "rust", "python", "java"]
    outputs = {}
    for lang in languages:
        out_file = test_dir / f"golden_output_{lang}.json"
        assert out_file.exists(), f"Missing output file: {out_file}"
        with open(out_file, "r", encoding="utf-8") as f:
            outputs[lang] = json.load(f)

    print("\n--- Verifying Bit-Exact Structural Hash & Numerical Parity ---")
    hashes = {lang: int(outputs[lang]["structural_hash"]) for lang in languages}
    values = {lang: float(outputs[lang]["value_estimate"]) for lang in languages}
    policies = {lang: np.array(outputs[lang]["policy_logits"], dtype=np.float32) for lang in languages}
    latents = {lang: np.array(outputs[lang]["latent_embedding"], dtype=np.float32) for lang in languages}

    # 1. Structural Hash Exact Match
    ref_hash = hashes["cpp"]
    for lang, h in hashes.items():
        print(f"[{lang.upper()}] Structural Hash: {h} (0x{h:016x})")
        assert h == ref_hash, f"Structural hash mismatch in {lang}: {h} != {ref_hash}"
    print(">>> PASS: All 4 languages computed IDENTICAL 64-bit structural hash!\n")

    # 2. Value Estimate Numerical Parity
    ref_val = values["cpp"]
    EPSILON = 1e-6
    for lang, v in values.items():
        diff = abs(v - ref_val)
        print(f"[{lang.upper()}] Value Estimate: {v:.7f} (diff vs cpp: {diff:.2e})")
        assert diff <= EPSILON, f"Value estimate diff {diff} exceeds {EPSILON} for {lang}"
    print(">>> PASS: All 4 languages match Value Estimate within eps <= 1e-6!\n")

    # 3. Policy Logits Max Absolute Difference
    ref_policy = policies["cpp"]
    assert ref_policy.size == 4 * 16, f"Expected 64 elements, got {ref_policy.size}"
    for lang, p in policies.items():
        max_diff = float(np.max(np.abs(p - ref_policy)))
        print(f"[{lang.upper()}] Policy Logits max abs diff vs cpp: {max_diff:.2e}")
        assert max_diff <= EPSILON, f"Policy logits max diff {max_diff} exceeds {EPSILON} for {lang}"
    print(">>> PASS: All 4 languages match Policy Logits within eps <= 1e-6!\n")

    # 4. Latent Embedding Max Absolute Difference
    ref_latent = latents["cpp"]
    assert ref_latent.size == 64, f"Expected 64 elements, got {ref_latent.size}"
    for lang, l in latents.items():
        max_diff = float(np.max(np.abs(l - ref_latent)))
        print(f"[{lang.upper()}] Latent Embedding max abs diff vs cpp: {max_diff:.2e}")
        assert max_diff <= EPSILON, f"Latent embedding max diff {max_diff} exceeds {EPSILON} for {lang}"
    print(">>> PASS: All 4 languages match Latent Embeddings within eps <= 1e-6!\n")

    print("=================================================================")
    print(" >>> CROSS-LANGUAGE NUMERICAL PARITY FULLY VERIFIED (100%) <<<  ")
    print("=================================================================")

if __name__ == "__main__":
    test_cross_language_numerical_identity()
