# setup.py
from setuptools import setup, find_packages

setup(
    name="rl4mip",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "pyscipopt",
    ],
)