# python-project-template

A project template for pure python projects.

## Usage of this repository

This repository is at the moment setup as an example repository,
not as e.g. a cookie-cutter template that enables automated creation
of new repositories that follow this template.
This makes testing this repository easier for now, we will investigate
turning this repository into a proper cookie-cutter template in the future.

For now, to setup a repository, you need to copy / rename / adapt the files
in this repository:
* Change the name of the package and other metadata in `pyproject.toml` and `docs/conf.py`
* Change the the package directory to correspond to your new package in `src/<package name>`
* Setup the necessary CI Variables:
    * `DOCS_ACCESS_TOKEN` needs to be a gitlab personal access token with `read_api` permissions
    * `TWINE_PASSWORD` needs to be a pypi token, preferably one with a scope only for your project.
    This requires a manual first upload to (test.)pypi though.
    * `SONAR_TOKEN`, this variable is set when following the repository setup instructions in sonarqube


## Project Structure / Packaging

This project uses `setuptools` for packaging and defines all necessary options
in the file `pyproject.toml`. `setup.py` and`setup.cfg` are not needed.

* `setuptools` documentation: https://setuptools.pypa.io/en/latest/userguide/index.html
* `Configuring setuptools with pyprojet.toml`: https://setuptools.pypa.io/en/latest/userguide/pyproject_config.html
* Relevant PEPs:
    * [PEP 517 – A build-system independent format for source trees](https://peps.python.org/pep-0517)
    * [PEP 518 – Specifying Minimum Build System Requirements for Python Projects](https://peps.python.org/pep-0518)
    * [PEP 621 – Storing project metadata in pyproject.toml](https://peps.python.org/pep-0621/)
    * [PEP 660 – Editable installs for pyproject.toml based builds (wheel based)](https://peps.python.org/pep-0660/)

We use the `src/` based layout, as this avoids several issues with editable installs and confusion with what
is imported (local directory or installed module).
See [setuptools/src-layout](https://setuptools.pypa.io/en/latest/userguide/package_discovery.html#src-layout).


### Editable installations

Editable installations in this setup  rely on PEP 660 (see above), support was introduced in `pip` 21.3 (released 2021-10)
and setuptools 64.0 (released 2022-08). The setuptools version is required in `pyproject.toml`.

To install in editable mode, use
```
$ pip install -e .
```

you can add extras, e.g. for developing and building the docs, use
```
$ pip install -e '.[dev,doc,test]'
```

or just
```
$ pip install -e '.[all]'
```


Keep in mind that editable installations have limitations as to what changes can take effect automatically
without rerunning `pip install -e .`. Python code changes to existing files take effect, but for example
adding new entry-points, changes to the source code of compiled extensions etc. will require rerunning the
installation.

See <https://setuptools.pypa.io/en/latest/userguide/development_mode.html#limitations>

## Versioning

This template uses `setuptools_scm` to automatically generate the version from the last git tag.
For local development, `setuptools_scm` will build a version development version. For releases,
version information is included in the sdist and wheel files and `setuptools_scm` is not used
when installing those. The setup is somewhat complex and was taken by astropy. It ensures that
the setuptools scm version is only used in development setups, not in the releases packages.

## CI

For github, see `.github/workflows/`, for gitlab, see `.gitlab-ci.yml`


## Integration with the CTAO Sonarqube instances

The `.gitlab-ci.yaml` contains a step to scan the project using sonarqube,
also ingesting the coverage report created in the previous step.
This needs the `SONAR_TOKEN` secret variable to be set in the CI configuration
of the repository and adapting the configuration in `sonar-project.properties`
for the given project.
Token and project key can be obtained when creating a new project in the sonarqube
instance via the GitLab integration, just follow the steps.

## Docs

Build the documentation in html format using:

```
$ make -C docs html
```

To view them, you can run:

```
$ python -m http.server -d docs/build/html
```

You can also install `sphinx-autobuild` and run

```
sphinx-autobuild docs docs/build/html
```
to get a continuously running and updating preview of the docs while you edit them.

Docs are automatically deployed on main and tags to gitlab pages.
