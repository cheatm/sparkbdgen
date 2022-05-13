from setuptools import setup, find_packages
from sparkbdgen import __version__


pkgs = find_packages()


setup(
    name="sparkbdgen",
    version=__version__,
    description="Big data generator for spark.",
    packages=pkgs,

)