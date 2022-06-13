import os
from setuptools import setup, find_packages


version = None
with open(os.path.join('digipd_ml', '__init__.py'), 'r') as fid:
    for line in (line.strip() for line in fid):
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().strip('\'')
            break
if version is None:
    raise RuntimeError('Could not determine version')

DISTNAME = 'digipd_ml'
DESCRIPTION = 'NestedCV for classification and regression'
with open('README.rst', 'r') as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = 'Quentin Klopfenstein'
MAINTAINER_EMAIL = 'quentin.klopfenstein@uni.lu'
LICENSE = 'BSD (3-clause)'
DOWNLOAD_URL = 'https://github.com/Klopfe/digipd_ml'
VERSION = version

setup(name=DISTNAME,
      version=version,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      download_url=DOWNLOAD_URL,
      packages=find_packages(),
      install_requires=['numpy>=1.12', 'scikit-learn>=1.0', 'imodels', 'pandas']
      )