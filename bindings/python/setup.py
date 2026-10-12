from setuptools import setup, find_packages

setup(
    name="axman",
    version="1.0.0",
    description="Python bindings for AXMANet (Axiomatic eXploration & Morphism Architecture Network)",
    author="Project Makora / AXMA Authors",
    license="Apache-2.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20",
    ],
    extras_require={
        "torch": ["torch>=2.0.0"],
    },
    python_requires=">=3.9",
)
