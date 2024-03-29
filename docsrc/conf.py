# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys, os
sys.path.insert(0, os.path.abspath('..'))
from waveguicsx import __version__

project = 'waveguicsx'
copyright = '2023-2024, Fabien Treyssede'
author = 'Fabien Treyssede'
release = __version__

rst_epilog = '.. |__version__| replace:: %s' % __version__

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
# html_static_path = ['_static']
html_logo = "logo_doc.png"
html_theme_options = {
    # 'analytics_id': 'G-XXXXXXXXXX',  #  Provided by Google in your dashboard
    # 'analytics_anonymize_ip': False,
    'logo_only': True,
    'display_version': False,
    # 'prev_next_buttons_location': 'bottom',
    # 'style_external_links': False,
    # 'vcs_pageview_mode': '',
    # 'style_nav_header_background': 'white',
    # Toc options
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 2,
    'includehidden': True,
    'titles_only': False
    }
