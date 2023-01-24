# Copyright 2021-2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Package setup."""

import os
import shutil
import subprocess
import setuptools
from setuptools import find_packages
from setuptools import setup
from setuptools.command import build_ext

with open("README.md", "r") as f:
  long_description = f.read()

with open("requirements.txt", "r") as f:
  dependencies = list(map(lambda x: x.strip(), f.readlines()))


class CMakeExtension(setuptools.Extension):
  """A Python extension that has been prebuilt by CMake.

  We do not want distutils to handle the build process for our extensions, so
  so we pass an empty list to the super constructor.
  """

  def __init__(self, name):
    super().__init__(name, sources=[])


class BuildCMakeExtension(build_ext.build_ext):
  """Uses CMake to build extensions."""

  def run(self):
    self._build()
    for ext in self.extensions:
      self.build_extension(ext)

  def _build(self):
    print("Building C++ extension")
    os.makedirs(self.build_temp, exist_ok=True)
    subprocess.check_call(
        ["cmake"]
        + [os.path.join(os.getcwd(), "transformer_grammars/models/masking")],
        cwd=self.build_temp,
    )
    subprocess.check_call(
        ["cmake", "--build", ".", "--", "-j"], cwd=self.build_temp
    )

  def build_extension(self, ext):
    dest_path = self.get_ext_fullpath(ext.name)
    build_path = os.path.join(self.build_temp, os.path.basename(dest_path))
    shutil.copyfile(build_path, dest_path)


setup(
    name="transformer_grammars",
    version="1.0.0",
    url="https://github.com/deepmind/transformer_grammars",
    author="Laurent Sartran et al.",
    author_email="lsartran@deepmind.com",
    description="Implementation of Transformer Grammars",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=dependencies,
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    cmdclass=dict(build_ext=BuildCMakeExtension),
    ext_modules=[
        CMakeExtension("transformer_grammars.models.masking.cpp_masking"),
    ],
    python_requires=">=3.7",
    test_suite="transformer_grammars",
    dependency_links=[
        "https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
    ],
)
