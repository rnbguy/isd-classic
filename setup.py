# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


requirements = [
    "numpy>=1.15",
]

setup(
    name="isdclassic",
    version="0.1.0",
    description="Software that implements a few ISD algorithms",
    url="https://github.com/tigerjack/isd-prange",
    author="tigerjack",
    classifiers=[
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering",
    ],
    keywords="isd prange",
    packages=find_packages(exclude=['test*', 'experiments*']),
    install_requires=requirements,
    include_package_data=True,
    python_requires=">=3.5",
)
