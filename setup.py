import setuptools
import os

setuptools.setup(
    name='waveguicsx',
    author="Fabien Treyssede",
    # author_email="", TODO add mail adress if useful
    version="0.0",
    packages=setuptools.find_packages(),
    url='https://github.com/treyssede/waveguicsx',
    # license='LICENCE',  TODO: add a licence file
    # description='', TODO add a short description
    # long_description='', # possibly the loaded content of README.md
    # long_description_content_type="text/markdown", # if README.md
    scripts=[],  # TODO add list of callable python scripts here
    classifiers=[
        "Programming Language :: Python :: 3"
        "Operating System :: Linux",
        ],
    install_requires=[], # list the python packages to install with python -m pip install
    python_requires='>=3')
