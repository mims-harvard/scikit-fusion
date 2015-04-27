#! /usr/bin/env python

descr = """A Python module for data fusion built on top of factorized models."""

import sys
import os

DISTNAME = 'scikit-fusion'
DESCRIPTION = 'A Python module for data fusion built on top of factorized models.'
LONG_DESCRIPTION = open('README.rst').read()
MAINTAINER = 'Marinka Zitnik'
MAINTAINER_EMAIL = 'marinka.zitnik@fri.uni-lj.si'
URL = 'http://xyz.biolab.si'
LICENSE = 'GPL3'
VERSION = '0.1'


###############################################################################
def configuration(parent_package='', top_path=None):
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)

    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)

    config.add_subpackage('skfusion')

    return config


def setup_package():
    metadata = dict(name=DISTNAME,
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
                                 'Programming Language :: Python :: 2.6',
                                 ],)

    if (len(sys.argv) >= 2
            and ('--help' in sys.argv[1:] or sys.argv[1]
                 in ('--help-commands', 'egg_info', '--version', 'clean', 'develop'))):
        try:
            from setuptools import setup
        except ImportError:
            from distutils.core import setup

        metadata['version'] = VERSION
    else:
        from numpy.distutils.core import setup

        metadata['configuration'] = configuration

    setup(**metadata)


if __name__ == "__main__":
    setup_package()
