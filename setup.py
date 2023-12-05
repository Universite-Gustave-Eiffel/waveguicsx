import setuptools
import os
import subprocess


class MakeTheDoc(setuptools.Command):
    description = "Generate Documentation Pages using Sphinx"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        """The command to run when users invoke python setup.py doc"""
        subprocess.run(
            ['sphinx-build docsrc docs'], shell=True)
        

with open('README.md', 'r') as fid:
    long_description = fid.read()


setuptools.setup(
    name='waveguicsx',
    author="Fabien Treyssede",
    # author_email="", TODO add mail adress if useful
    version="0.0",
    packages=setuptools.find_packages(),
    url='https://github.com/treyssede/waveguicsx',
    license='COPYING',
    description='waveguicsx, a python library for solving complex waveguide problems', 
    long_description=long_description, 
    long_description_content_type="text/markdown",
    scripts=[],  # TODO add list of callable python scripts here
    classifiers=[
        "Programming Language :: Python :: 3"
        "Operating System :: Linux",
        ],
    cmdclass={
        'doc': MakeTheDoc,  # allow user to build the doc with python setup.py doc
        },
    install_requires=[
        # required packages
        'numpy', 'matplotlib',
        'petsc4py', 'slepc4py', 'mpi4py',  # <- fail with pip, use conda -c conda-forge ... see readme
        # packages useful for the tutorials
        'pyvista',
        # packages required for sphinx
        'sphinx', 'sphinx-rtd-theme', 'myst-parser',  'nbsphinx',
        ],  # list the python packages to install with python -m pip install
    python_requires='>=3')
