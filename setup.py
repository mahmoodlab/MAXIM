from setuptools import setup, find_packages

PYPI_REQUIREMENTS = [
    "torch==1.13.1",
    "torchvision==0.14.1",
    "numpy>=1.21.5",
    "pandas>=1.4.4",
    "scikit_image>=0.19.2",
    "scipy>=1.9.1",
    "captum>=0.6.0"
]
setup(
    name='maxim',  # A string containing the package’s name.
    version='1.0.0',  # A string containing the package’s version number.
    description='Marker imputation model for multiplex images.',  # A single-line text explaining the package.
    long_description='',  # A string containing a more detailed description of the package.

    maintainer='',  # It's a string providing the current maintainer’s name, if not the author.
    url='https://github.com/mahmoodlab/MAXIM/',  # A string providing the package’s homepage URL (usually the GitHub repository or the PyPI page).
    download_url='https://github.com/mahmoodlab/MAXIM/',  # A string containing the URL where the package may be downloaded.
    author='Muhammad Shaban',  # A string identifying the package’s creator/author.
    author_email='muhammadshaban.cs@gmail.com',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Custom Apache 2.0 with Commons Clause License",
        "Operating System :: OS Independent",
    ],

    packages= find_packages('.'),

    # A string list containing only the dependencies necessary for the package to function effectively.
    install_requires=PYPI_REQUIREMENTS,
)
