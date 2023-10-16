# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys, os
sys.path.insert(0, os.path.abspath('..'))

project = 'waveguicsx'
copyright = '2023, Fabien Treyssede'
author = 'Fabien Treyssede'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.todo", "sphinx.ext.viewcode", "sphinx.ext.autodoc",
              'sphinx.ext.mathjax', 'sphinx.ext.ifconfig',
              "myst_parser",
              'nbsphinx']

# myst-parser is used to manage (readme) markdown file
# ref : https://myst-parser.readthedocs.io/en/latest/syntax/optional.html
myst_enable_extensions = [
    "amsmath",
#    "attrs_inline",
#    "colon_fence",
#    "deflist",
    "dollarmath",   # to allow latex from markdown to be rendered in sphinx
#    "fieldlist",
#    "html_admonition",
#    "html_image",
#    "linkify",
#    "replacements",
#    "smartquotes",
#    "strikethrough",
#    "substitution",
#    "tasklist",
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
autodoc_member_order = 'bysource'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
