from pathlib import Path
import setuptools
from setuptools import setup

required = Path("requirements.txt").read_text(encoding="utf-8").split("\n")

setup(name='proto_clip_toolkit', 
      description="A simple toolkit from Proto-CLIP demo that provies speech recognition, part-of-speech tagging and realworld robot demo APIs.",
      author="IRVL-UTd: Intelligent Robotics and Vision Lab at the University of Texas at Dallas",
      author_email="irvl.utd@gmail.com",
      version='0.1',
      url="https://github.com/IRVLUTD/Proto-CLIP",
      license="MIT",
      packages=setuptools.find_packages(),
      install_requires=required,
      include_package_data=True,
      python_requires=">=3.7"
)
