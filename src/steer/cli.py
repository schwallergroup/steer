# -*- coding: utf-8 -*-

"""Command line interface for :mod:`steer`.

Why does this file exist, and why not put this in ``__main__``? You might be tempted to import things from ``__main__``
later, but that will cause problems--the code will get executed twice:

- When you run ``python3 -m steer`` python will execute``__main__.py`` as a script.
  That means there won't be any ``steer.__main__`` in ``sys.modules``.
- When you import __main__ it will get executed again (as a module) because
  there's no ``steer.__main__`` in ``sys.modules``.

.. seealso:: https://click.palletsprojects.com/en/8.1.x/setuptools/#setuptools-integration
"""

import logging

import click

__all__ = [
    "main",
]

logger = logging.getLogger(__name__)


@click.group()
@click.version_option()
def main():
    """CLI for steer."""


if __name__ == "__main__":
    main()
