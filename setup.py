#! /usr/bin/env python
descr = """A Python module for data fusion built on top of factorized models."""

from setuptools import setup, find_packages
import subprocess
import os
import imp

DISTNAME = 'scikit-fusion'
DESCRIPTION = 'A Python module for data fusion built on top of factorized models.'
LONG_DESCRIPTION = open('README.md').read()
MAINTAINER = 'Marinka Zitnik'
MAINTAINER_EMAIL = 'marinka@cs.stanford.edu'
URL = 'http://github.com/marinkaz/scikit-fusion'
LICENSE = 'GPLv3'
VERSION = '0.2.1'
ISRELEASED = False

INSTALL_REQUIRES = (
    'numpy>=1.7.0',
    'scipy>=0.12.0',
    'joblib>=0.8.4',
)

# Return the git revision as a string
def git_version():
    """Return the git revision as a string.

    Copied from numpy setup.py
    """
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(cmd, stdout = subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        GIT_REVISION = out.strip().decode('ascii')
    except OSError:
        GIT_REVISION = "Unknown"
    return GIT_REVISION


def write_version_py(filename='skfusion/version.py'):
    # Copied from numpy setup.py
    cnt = """
# THIS FILE IS GENERATED FROM SCIKIT-FUSION SETUP.PY
short_version = '%(version)s'
version = '%(version)s'
full_version = '%(full_version)s'
git_revision = '%(git_revision)s'
release = %(isrelease)s

if not release:
    version = full_version
    short_version += ".dev"
"""
    FULLVERSION = VERSION
    if os.path.exists('.git'):
        GIT_REVISION = git_version()
    elif os.path.exists('skfusion/version.py'):
        # must be a source distribution, use existing version file
        version = imp.load_source("skfusion.version", "skfusion/version.py")
        GIT_REVISION = version.git_revision
    else:
        GIT_REVISION = "Unknown"

    if not ISRELEASED:
        FULLVERSION += '.dev0+' + GIT_REVISION[:7]

    a = open(filename, 'w')
    try:
        a.write(cnt % {'version': VERSION,
                       'full_version': FULLVERSION,
                       'git_revision': GIT_REVISION,
                       'isrelease': str(ISRELEASED)})
    finally:
        a.close()


def setup_package():
    write_version_py()
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
