# -*- coding: utf-8 -*-

"""Main code."""

import asyncio
import json
import logging
import os

import click

from steer.logger import setup_logger

logger = setup_logger(__name__)
