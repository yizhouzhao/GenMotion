# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'GenMotion'
copyright = '2021, UCLA'
author = 'xxx'

release = '0.1'
version = '0.1.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'nbsphinx',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join('..', '..', 'src')))

autodoc_mock_imports = ["bpy", "numpy", "torch", "torch.nn", "torch.nn.functional"]