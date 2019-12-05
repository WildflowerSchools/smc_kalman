import os
from setuptools import setup, find_packages

BASEDIR = os.path.dirname(os.path.abspath(__file__))
VERSION = open(os.path.join(BASEDIR, 'VERSION')).read().strip()

BASE_DEPENDENCIES = [
    'numpy>=1.15',
    'scipy>=1.1',
    'matplotlib>=2.2'
]

# allow setup.py to be run from any path
os.chdir(os.path.normpath(BASEDIR))

setup(
    name='wf-smc-kalman',
    packages=find_packages(),
    version=VERSION,
    include_package_data=True,
    description='Sequential Monte Carlo modeling for linear Gaussian systems',
    long_description=open('README.md').read(),
    url='https://github.com/WildflowerSchools/smc_kalman',
    author='Theodore Quinn',
    author_email='ted.quinn@wildflowerschools.org',
    install_requires=BASE_DEPENDENCIES,
    keywords=['smc'],
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
    ]
)
