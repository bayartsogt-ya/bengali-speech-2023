"""setup.py for `bengali_speech` package."""

import os
import setuptools
from setuptools import find_packages

setuptools.setup(
    name='bengali_speech',
    version=open('version.txt', 'r').read().strip() if os.path.isfile('version.txt') else '0.0.1',
    author='Bayartsogt Yadamsuren',
    author_email='bayartsogt.yadamsuren@gmail.com',
    description='',
    long_description=open('README.md', 'r').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/bayartsogt-ya/bengali-speech-2023',
    classifiers=[
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.10',
    ],
    package_dir={'': 'src'},
    packages=find_packages(where='src', exclude=['tests']),
    install_requires=[],
    python_requires='>=3.10',
)
