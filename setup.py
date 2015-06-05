#! /usr/bin/env python
descr = """A Python module for data fusion built on top of factorized models."""

from setuptools import setup, find_packages

DISTNAME = 'scikit-fusion'
DESCRIPTION = 'A Python module for data fusion built on top of factorized models.'
LONG_DESCRIPTION = open('README.rst').read()
MAINTAINER = 'Marinka Zitnik'
MAINTAINER_EMAIL = 'marinka.zitnik@fri.uni-lj.si'
URL = 'http://xyz.biolab.si'
LICENSE = 'GPLv3'
VERSION = '0.1'

INSTALL_REQUIRES = (
    'numpy>=1.7.0',
    'scipy>=0.12.0',
    'joblib>=0.8.4',
)


def setup_package():
    setup(name=DISTNAME,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          license=LICENSE,
          url=URL,
          version=VERSION,
          long_description=LONG_DESCRIPTION,
          classifiers=['Intended Audience :: Science/Research',
                       'Intended Audience :: Developers',
                       'License :: OSI Approved',
                       'Programming Language :: Python',
                       'Topic :: Software Development',
                       'Topic :: Scientific/Engineering',
                       'Topic :: Scientific/Engineering :: Artificial Intelligence',
                       'Topic :: Scientific/Engineering :: Bio-Informatics',
                       'Programming Language :: Python :: 2',
                       'Programming Language :: Python :: 3',
                       ],
          install_requires=INSTALL_REQUIRES,
          packages=find_packages(),
          package_data={
              'skfusion.datasets': ['data/*/*']
          },
          test_suite='skfusion.tests.test_suite',
)


if __name__ == "__main__":
    setup_package()
