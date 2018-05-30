#!/usr/bin/env python

"""A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

"""A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path


setup(
    name='h3mlcore',
    version='0.82dev',
    url='https://github.com/CognutSecurity/h3mlcore',
    packages=find_packages(exclude=['test']),
    author='Huang Xiao',
    author_email='xh0217@gmail.com',
    license='Creative Commons license',
    long_description=open('README.md').read(),
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',

        # Pick your license as you wish (should match "license" above)

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
    ],

    install_requires=[
        "setuptools == 36.5.0",
        "nltk == 3.2.1",
        "mxnet == 0.12.0",
        "paramiko == 2.0.2",
        "pandas == 0.18.1",
        "termcolor == 1.1.0",
        "scipy == 0.18.0",
        "causality == 0.0.3",
        "dill == 0.2.5",
        "simplejson == 3.11.1",
        "libpgm == 1.3",
        "bokeh == 0.12.15",
        "numpy == 1.13.1",
        "matplotlib == 1.5.1",
        "javalang == 0.11.0",
        "scikit_learn == 0.19.1",
        "PyYAML == 3.12",
    ],

    python_requires='>=2.6, !=3.*',

    keywords='machine learning data mining',

)
