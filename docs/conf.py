# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Add the source directory to the path for autodoc
sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -----------------------------------------------------
project = "Black Hole Video Evaluation"
copyright = "2025, Rohan Dahale"
author = "Rohan Dahale"
version = "0.1.0"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
# Note: autodoc and autosummary are disabled until the package is properly
# structured and dependencies are available in CI. Enable them later by
# uncommenting the lines below.
extensions = [
    # "sphinx.ext.autodoc",         # Auto-generate docs from docstrings
    # "sphinx.ext.autosummary",     # Generate summary tables
    # "sphinx.ext.napoleon",        # Support NumPy/Google style docstrings
    # "sphinx.ext.viewcode",        # Add links to source code
    "sphinx.ext.intersphinx",       # Link to other projects' docs
    "sphinx.ext.mathjax",           # Render LaTeX math
    "sphinx_copybutton",            # Copy button for code blocks
]

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_typehints = "description"
autosummary_generate = True

# Napoleon settings (for NumPy-style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
}

# Templates and exclusions
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Source file settings
source_suffix = ".rst"
master_doc = "index"

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"

# Furo theme options - Beautiful dark/light theme
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#7C4DFF",      # Purple accent
        "color-brand-content": "#7C4DFF",
        "color-admonition-background": "#f8f9fa",
    },
    "dark_css_variables": {
        "color-brand-primary": "#B388FF",      # Light purple for dark mode
        "color-brand-content": "#B388FF",
        "color-background-primary": "#0d1117",
        "color-background-secondary": "#161b22",
        "color-admonition-background": "#1c2128",
    },
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "top_of_page_button": "edit",
    "source_repository": "https://github.com/rohandahale/ehteval",
    "source_branch": "main",
    "source_directory": "docs/",
}

html_title = "Black Hole Video Evaluation"
html_short_title = "Black Hole Video Evaluation"

# Static files (custom CSS, images, etc.)
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# Favicon and logo (optional - add your own!)
# html_logo = "_static/logo.png"
# html_favicon = "_static/favicon.ico"

# -- MathJax configuration ---------------------------------------------------
mathjax3_config = {
    "tex": {
        "macros": {
            "RR": r"\mathbb{R}",
            "bold": [r"\mathbf{#1}", 1],
        }
    }
}

# -- Other options -----------------------------------------------------------
# Show "Created using Sphinx" in the HTML footer
html_show_sphinx = True
html_show_copyright = True

# Pygments style for syntax highlighting
pygments_style = "friendly"
pygments_dark_style = "monokai"
