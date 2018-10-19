import setuptools

setuptools.setup(
    name="smc_kalman",
    version="0.0.1",
    author="Theodore Quinn",
    author_email="ted.quinn@wildflowerschools.org",
    license='MIT',
    description="Sequential Monte Carlo modeling for linear Gaussian systems",
    url="https://github.com/WildflowerSchools/smc_kalman",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy>=1.15',
        'scipy>=1.1',
        'matplotlib>=2.2'],
    classifiers=(
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent"))
