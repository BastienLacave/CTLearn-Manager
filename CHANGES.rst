template v0.9.0 (2024-11-13)
----------------------------

New Features
~~~~~~~~~~~~

- Use the newly released ``ctao-sphinx-theme`` for the documentation.

  To use it, apply the following changes in ``docs/conf.py``:

  .. code-block:: diff

     -html_theme = "pydata_sphinx_theme"
     +html_theme = "ctao"
      html_theme_options = dict(
          navigation_with_keys=False,
     -    logo=dict(
     -        image_light="_static/cta.png",
     -        image_dark="_static/cta_dark.png",
     -        alt_text="ctao-logo",
     -    ),
     )

  and replace ``pydata-sphinx-theme`` with ``ctao-sphinx-theme`` in the ``doc`` optional dependencies.

  [`!29 <https://gitlab.cta-observatory.org/cta-computing/documentation/python-project-template/-/merge_requests/29>`__]


template v0.8.0 (2024-07-10)
----------------------------

New Features
~~~~~~~~~~~~

- Add towncrier configuration for automatic changelog generation.

  The following files have been changed:

  - ``pyproject.toml``: Check the new ``[tool.towncrier]`` section and new dependency ``sphinx-changelog`` in the
    ``doc`` optional-dependencies.
  - ``docs/changes/README.md`` explaining the procedure.
  - ``docs/conf.py``, new extension ``sphinx_changelog``
  - ``docs/changelog.rst``, also now linked in ``docs/index.rst`` [`!27 <https://gitlab.cta-observatory.org/cta-computing/documentation/python-project-template/-/merge_requests/27>`__]


template v0.7.0 (2024-05-14)
----------------------------

- Switch to ruff / ruff format instead of pyflakes / pycodestyle / black
- Make sure pre-commit hooks do not modify fits files
- Do not turn numpy binary size changed warning into an error in pytest
- Drop support for python 3.9
- Add code-spell, repo-check and standard pre-commit hooks
