import setuptools
import os
import subprocess

# ================================= 
# Installation folder
WAVEGUICSXHOME = os.path.dirname(__file__)

# =================================
# Add command to build the doc from sources with
# python setup.py doc
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
        
# =================================
# Get the version variable 
def read_version_number():
    # read the variable manually, skip distutils2-boolshit
    version_file = os.path.join(WAVEGUICSXHOME, 'waveguicsx', 'version.py')
    if not os.path.isfile(version_file):
        raise IOError(version_file)

    with open(version_file, "r") as fid:
        for line in fid:
            if line.strip('\n').strip().startswith('__version__'):
                version_number = line.strip('\n').split('=')[-1].split()[0].strip().strip('"').strip("'")
                break
        else:
            raise Exception(f'could not detect __version__ affectation in {version_file}')
    return version_number
    
__version__ = read_version_number()


# =================================
# Load the README Content
with open('README.md', 'r') as fid:
    long_description = fid.read()

# =================================
# Install the package
setuptools.setup(
    name='waveguicsx',
    author="Fabien Treyssede",
    # author_email="", TODO add mail adress if useful
    version=__version__,
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
        'numpy', 'matplotlib',  # 'pyvista, 'scipy',... ?
        'petsc4py', 'slepc4py',
        # packages required for sphinx
        'sphinx', 'sphinx-rtd-theme', 'myst-parser',  'nbsphinx',
        ], # list the python packages to install with python -m pip install
    python_requires='>=3')
