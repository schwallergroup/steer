


<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./assets/repo_logo_dark.png" width='100%'>
  <source media="(prefers-color-scheme: light)" srcset="./assets/repo_logo_light.png" width='100%'>
  <img alt="Project logo" src="/assets/" width="100%">
</picture>

<br>

[![tests](https://github.com/schwallergroup/steer/actions/workflows/tests.yml/badge.svg)](https://github.com/schwallergroup/steer)
[![DOI:10.1101/2020.07.15.204701](https://zenodo.org/badge/DOI/10.48550/arXiv.2304.05376.svg)](https://doi.org/10.48550/arXiv.2304.05376)
[![PyPI](https://img.shields.io/pypi/v/steer)](https://img.shields.io/pypi/v/steer)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/steer)](https://img.shields.io/pypi/pyversions/steer)
[![Documentation Status](https://readthedocs.org/projects/steer/badge/?version=latest)](https://steer.readthedocs.io/en/latest/?badge=latest)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Cookiecutter template from @SchwallerGroup](https://img.shields.io/badge/Cookiecutter-schwallergroup-blue)](https://github.com/schwallergroup/liac-repo)
[![Learn more @SchwallerGroup](https://img.shields.io/badge/Learn%20%0Amore-schwallergroup-blue)](https://schwallergroup.github.io)




<h1 align="center">
  steer
</h1>


<br>

LLM-guided search of chemical instances


## 🔥 Usage

We handle 2 use-cases for the moment:


### Retrosynthetic planning

Run the synthesis reranking benchmark

> steer synth all-task

Run a single task

> steer synth one-task

### Mechanism finding

Run benchmark on step-by-step selection

> steer mech bench


## 👩‍💻 Installation

<!-- Uncomment this section after your first ``tox -e finish``
The most recent release can be installed from
[PyPI](https://pypi.org/project/steer/) with:

```shell
$ pip install steer
```
-->

The most recent code and data can be installed directly from GitHub with:

```bash
$ pip install git+https://github.com/schwallergroup/steer.git
```

## ✅ Citation

Philippe Schwaller et al. "Molecular Transformer: A Model for Uncertainty-Calibrated Chemical Reaction Prediction". ACS Central Science 2019 5 (9), 1572-1583
```bibtex
@article{doi:10.1021/acscentsci.9b00576,
    author = {Schwaller, Philippe and Laino, Teodoro and Gaudin, Théophile and Bolgar, Peter and Hunter, Christopher A. and Bekas, Costas and Lee, Alpha A.},
    title = {Molecular Transformer: A Model for Uncertainty-Calibrated Chemical Reaction Prediction},
    journal = {ACS Central Science},
    volume = {5},
    number = {9},
    pages = {1572-1583},
    year = {2019},
    doi = {10.1021/acscentsci.9b00576},
}

@Misc{this_repo,
  author = { Andres M Bran },
  title = { steer - Steerable retrosynthesis with LLM },
  howpublished = {Github},
  year = {2023},
  url = {https://github.com/schwallergroup/steer }
}
```


## 🛠️ For Developers


<details>
  <summary>See developer instructions</summary>



### 👐 Contributing

Contributions, whether filing an issue, making a pull request, or forking, are appreciated. See
[CONTRIBUTING.md](https://github.com/schwallergroup/steer/blob/master/.github/CONTRIBUTING.md) for more information on getting involved.


### Development Installation

To install in development mode, use the following:

```bash
$ git clone git+https://github.com/schwallergroup/steer.git
$ cd steer
$ pip install -e .
```

### 🥼 Testing

After cloning the repository and installing `tox` with `pip install tox`, the unit tests in the `tests/` folder can be
run reproducibly with:

```shell
$ tox
```

Additionally, these tests are automatically re-run with each commit in a [GitHub Action](https://github.com/schwallergroup/steer/actions?query=workflow%3ATests).

### 📖 Building the Documentation

The documentation can be built locally using the following:

```shell
$ git clone git+https://github.com/schwallergroup/steer.git
$ cd steer
$ tox -e docs
$ open docs/build/html/index.html
```

The documentation automatically installs the package as well as the `docs`
extra specified in the [`setup.cfg`](setup.cfg). `sphinx` plugins
like `texext` can be added there. Additionally, they need to be added to the
`extensions` list in [`docs/source/conf.py`](docs/source/conf.py).

### 📦 Making a Release

After installing the package in development mode and installing
`tox` with `pip install tox`, the commands for making a new release are contained within the `finish` environment
in `tox.ini`. Run the following from the shell:

```shell
$ tox -e finish
```

This script does the following:

1. Uses [Bump2Version](https://github.com/c4urself/bump2version) to switch the version number in the `setup.cfg`,
   `src/steer/version.py`, and [`docs/source/conf.py`](docs/source/conf.py) to not have the `-dev` suffix
2. Packages the code in both a tar archive and a wheel using [`build`](https://github.com/pypa/build)
3. Uploads to PyPI using [`twine`](https://github.com/pypa/twine). Be sure to have a `.pypirc` file configured to avoid the need for manual input at this
   step
4. Push to GitHub. You'll need to make a release going with the commit where the version was bumped.
5. Bump the version to the next patch. If you made big changes and want to bump the version by minor, you can
   use `tox -e bumpversion -- minor` after.
</details>
