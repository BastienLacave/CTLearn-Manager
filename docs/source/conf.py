# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'CTLearn Manager'
copyright = '2025, Bastien Lacave'
author = 'Bastien Lacave'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',  # Automatically generate documentation from docstrings
    'sphinx.ext.napoleon', # Support for Google-style and NumPy-style docstrings
    'sphinx.ext.viewcode',  # Add links to source code
    'sphinx.ext.intersphinx',  # Link to other projects' documentation
    'sphinx.ext.todo', # Support for .. todo:: directives
    'sphinx.ext.coverage', # Measure documentation coverage
    'sphinx.ext.mathjax', # Render math equations
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']


autodoc_default_options = {
    'members': True,           # Include members (functions, classes, etc.)
    'member-order': 'bysource', # Order members as they appear in the source code
    'undoc-members': True,     # Include members *without* docstrings (useful during development)
    'private-members': False,   # Usually exclude private members (starting with _)
    'show-inheritance': True,   # Show class inheritance
}
autodoc_typehints = "description" # Show type hints in the description, not the signature.  Much cleaner.
autoclass_content = "both"  # Include both the class docstring and the __init__ docstring