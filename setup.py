#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from setuptools import setup

with open("README.md") as f:
    readme = f.read()

setup(
    name="distributed_faiss",
    version="0.0.1",
    description="Facebook AI Research Distributed Faiss",
    url="",  # TODO
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: CC-BY-NC",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    long_description=readme,
    long_description_content_type="text/markdown",
    setup_requires=["setuptools>=18.0"],
    install_requires=[
        "black",
        "cython",
        "faiss-cpu>=1.7.2",
        "filelock",
        "numpy",
        "regex",
        "submitit>=1.1.5",
        "torch>=1.2.0",
    ],
)
