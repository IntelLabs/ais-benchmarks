# Copyright 2023 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""setup.py for Brax.

Install for development:

  pip intall -e .
"""

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
