from pathlib import Path
import setuptools
from setuptools import setup

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()
required = (this_directory / "requirements.txt").read_text().split("\n")


setup(name='proto_clip_toolkit', 
      description="A simple toolkit from Proto-CLIP demo that provies speech recognition, part-of-speech tagging and realworld robot demo APIs.",
      author="IRVL-UTD: Intelligent Robotics and Vision Lab at the University of Texas at Dallas",
      author_email="irvl.utd@gmail.com",
      version='0.2',
      install_requires=required,
      url="https://github.com/IRVLUTD/Proto-CLIP",
      license="MIT",
      packages=setuptools.find_packages(),
      include_package_data=True,
      python_requires=">=3.7",
      long_description=long_description,
    long_description_content_type='text/markdown'
)
