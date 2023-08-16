"""
Configuration script for the Sphinx document builder

The intention is for this file to be used as a "global" configuration file
for any Python package to be documented using Sphinx so that all API
reference docs maintain a consistent look/feel/theme. Each project should
have a separate "sphinx.yaml" file that holds that project's information
and individual settings.

Provide a temporary environment variable pointing to the sphinx.conf file in
your project (relative to the location of this script) when you call
`sphinx-build`. For example:

```
CONFIG="../sphinx.yaml" sphinx-build -b html sphinx/ docs
```

This script is set to build for a given theme. See the HTML theme options
below.

For the full list of built-in configuration values, see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html

Author:
    Edge Impulse Inc.

Date:
    August 10, 2023

Copyright:
    Copyright 2023 Edge Impulse Inc.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""

import os
import sys
import importlib
from datetime import datetime

import m2r2
import yaml

###############################################################################
# Project Information
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-informationz

# Get the context (directory where we're executing)
context = os.environ.get('CONTEXT')
if context == None:
    context = "."
print(f"CONTEXT: {context}")

# Read in the configuration file
config_path = os.path.join(context, os.environ.get('CONFIG'))
print(f"CONFIG: {config_path}")
config_path = "sphinx.yaml" if config_path is None else config_path
try:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
except FileNotFoundError as e:
    print("ERROR: Config file not found")
    exit(1)

# Get project info from config file
project_name = config['project_info']['name']
package_name = config['project_info']['package_name']
package_path = config['project_info'].get('package_path', package_name)
package_path = os.path.join(context, package_path)

# Add location of package to path
sys.path.insert(0, os.path.abspath(package_path))

# Yes, we need to get the location of the package before importing it
package_handle = importlib.import_module(package_name)

# Generate copyright notice
copyright = str(datetime.now().year) + ", " + config['project_info']['author']

# Programmatically import the version from the package
version = importlib.metadata.version(package_name)
version = "" if version == "0.0.0" else version
print(f"Set version: {version}")

###############################################################################
# General Configuration 
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Read settings from configuration file
debug_callback_args = config.get('debug_callbacks', False)
readme_path = os.path.join(context, config.get('readme_path', "./README.md"))
source_path = config.get('source_path', "./source")
toc_maxdepth = config['toc'].get('maxdepth', -1)
toc_min_level = config['toc'].get('min_level', 1)
toc_max_level = config['toc'].get('max_level', 3)
toc_show_on_index = config['toc'].get('show_on_index', True)
exclude_suffixes = config.get('exclude_suffixes', [])
exclude_docstring_with_strings = config.get('exclude_docstring_with_strings', [])

# Include extensions
extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "myst_parser",
]

# Use templates (if they exist)
templates_path = []

# Exclude files/directories with the following patterns during processing
exclude_patterns = [
    "_build", 
    "Thumbs.db", 
    ".DS_Store",
    "modules.rst"
]

# Set language
language = "en"

# Allow for a signature to be broken into multiple lines
maximum_signature_line_length = 80

###############################################################################
# Napoleon Settings

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

###############################################################################
# HTML Theme Options
# https://pradyunsg.me/furo/

# Set theme
html_theme = "furo"

# Location of static assets (that do not change on each build)
html_static_path = ["assets"]

# Set favicon
html_favicon = config.get('favicon_path', None)

# Set title and version
html_title = f"{project_name}<br><font size='-0.5'>{version}</font>"

# Get HTML for the announcement banner
html_announcement_banner = None
if 'banner_path' in config:
    with open(config['banner_path'], 'r') as file:
        html_announcement_banner = file.read()

###############################################################################
# Theme customizations
# https://github.com/pradyunsg/furo/tree/main/src/furo/assets/styles/variables

html_theme_options = {

    # Logo
    "light_logo": "Edge Impulse primary logo - black text - 1400px.png",
    "dark_logo": "Edge Impulse primary logo - white text - 1400px.png",

    # Announcement banner
    "announcement": html_announcement_banner,

    # Customize colors and text for light mode
    "light_css_variables": {
        
        # Font
        "font-stack": "Arial, sans-serif",
        "font-stack--monospace": "Menlo, monospace",

        # Text colors
        "color-foreground-primary": "#333333",
        "color-foreground-secondary": "#333333",
        "color-brand-primary": "#333333",
        "color-brand-content": "#f6562c",

        # Announcement colors
        "color-announcement-background": "#00000000",
        "color-announcement-text": "#f6562c",

        # Sidebar colors
        "color-sidebar-background": "#00000000",
        "color-sidebar-search-background": "#00000000",

        # API colors
        "color-api-background": "#f8f9fb",
        "color-api-pre-name": "#333333",
        "color-api-name": "#935f00",
    },

    # Customize colors and text for dark mode
    "dark_css_variables": {

        # Text colors
        "color-foreground-primary": "#ffffffcc",
        "color-foreground-secondary": "#ffffffcc",
        "color-brand-primary": "#ffffffcc",
        "color-brand-content": "#f6562c",

        # Announcement colors
        "color-announcement-background": "#00000000",
        "color-announcement-text": "#f6562c",

        # Sidebar colors
        "color-sidebar-background": "#00000000",
        "color-sidebar-search-background": "#00000000",

        # API colors
        "color-api-background": "#202020",
        "color-api-pre-name": "#ffffffcc",
        "color-api-name": "#e99600",
    }
}

###############################################################################
# Generate index.rst
# Convert README.md to rST and append generated table of contents

# Read in the readme file
readme_md = ""
try:
    with open(os.path.join(context, readme_path), 'r') as f:
        readme_md = f.read()
except Exception as e:
    print(f"WARNING: Could not open readme file: {e}")
    readme_md = project_name + '\n' + ('=' * len(project_name))

# Convert readme to rST
readme_rst = m2r2.convert(readme_md)

# Get module exclude list
exclude_modules = config.get('exclude_modules', [])

# Construct list of modules for the TOC
module_sources = os.listdir(source_path)
toc_sources = []
for module_source in module_sources:
    module_name = module_source.split('.')[:-1]
    if (len(module_name) >= toc_min_level) and \
        (len(module_name) <= toc_max_level):
        module_name = '.'.join(module_name)
        if module_name not in exclude_modules:
            toc_sources.append(os.path.join(source_path, module_name))
toc_sources.sort()

# Construct index.rst text
index_text = """\
..
   Do not modify this file! It is automatically generated by conf.py.

"""
index_text += readme_rst
index_text += f"""\

.. toctree::
   :maxdepth: {toc_maxdepth}
"""
index_text += "" if toc_show_on_index else "   :hidden:\n"
index_text += """\
   :caption: Contents:

"""
for toc_source in toc_sources:
   index_text += f"   {toc_source}\n"
index_text +="""\

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
"""

# Write text to index.rst file
index_file_path = config.get('index_path', "./index.rst")
with open(index_file_path, 'w') as f:
    f.write(index_text)

###############################################################################
# Custom Callbacks
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#docstring-preprocessing

# Callback used to process docstrings
def autodoc_process_docstring_handler(app, what, name, obj, options, lines):

    # Print callback arguments
    if debug_callback_args:
        print("Docstring")
        print("app:", app)
        print("what:", what)
        print("name:", name)
        print("obj:", obj)
        print("options:", options)
        print("options.members", options['members'])
        print("lines:", lines)

    # Ignore docstring if exclude string is found
    for s in exclude_docstring_with_strings:
        if s in '\n'.join(lines):
            lines[:] = []
            if debug_callback_args:
                print("new lines:", lines)

    # Print line break
    if debug_callback_args:
        print("---")

# Callback prior to formatting a signature
def autodoc_before_process_signature_handler(app, obj, bound_method):

    # Print callback arguments
    if debug_callback_args:
        print("Before Signature")
        print("app:", app)
        print("obj:", obj)
        print("bound_method:", bound_method)
        print("---")

# Callback after formatting a signature
def autodoc_process_signature_handler(app, what, name, obj, options, signature, return_annotation):

    # Print callback arguments
    if debug_callback_args:
        print("After Signature")
        print("app:", app)
        print("what:", what)
        print("name:", name)
        print("obj:", obj)
        print("options:", options)
        print("signature:", signature)
        print("return_annotation:", return_annotation)
        print("---")

    return signature, return_annotation

# Callback after determining base class
def autodoc_process_bases_handler(app, name, obj, options, bases):

    # Print callback arguments
    if debug_callback_args:
        print("Bases")
        print("app:", app)
        print("name:", name)
        print("obj:", obj)
        print("options:", options)
        print("bases:", bases)
        print("---")

# Callback used to determine if we should skip a member
def autodoc_skip_member_handler(app, what, name, obj, skip, options):

    # Print callback arguments
    if debug_callback_args:
        print("Skip Member")
        print("app:", app)
        print("what:", what)
        print("name:", name)
        print("obj:", obj)
        print("skip:", skip)
        print("options:", options)
        print("---")

    # Skip member if name is marked with certain suffix
    if name.endswith(tuple(exclude_suffixes)):
        return True

    return skip

# Automatically called by sphinx at startup - assign callbacks
def setup(app):
    app.connect("autodoc-process-docstring", autodoc_process_docstring_handler)
    app.connect("autodoc-before-process-signature", autodoc_before_process_signature_handler)
    app.connect("autodoc-process-signature", autodoc_process_signature_handler)
    app.connect("autodoc-process-bases", autodoc_process_bases_handler)
    app.connect("autodoc-skip-member", autodoc_skip_member_handler)
    