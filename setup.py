# Copyright 2023 The AIS Benchmark Authors.


from setuptools import find_packages
from setuptools import setup

setup(
    name="ais_benchmarks",
    version="0.0.0",
    description=("A suite of sampling benchmarks"),
    author="Intel Labs",
    author_email="javier.felip.leon@intel.com",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/IntelLabs/ais-benchmarks",
    license="Apache 2.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "pillow",
        "pyyaml",
        "opencv-python",
        "guppy3",
    ],
    keywords="Probabilistic sampling benchmark",
)
