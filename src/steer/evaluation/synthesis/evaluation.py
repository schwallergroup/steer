"""Run evaluation script."""

import asyncio
import json
import os
from typing import List

import numpy as np

from steer.logger import setup_logger

from .eval_types import *
from .tasks import load_default_tasks

logger = setup_logger()

path = os.path.dirname(os.path.abspath(__file__))


def get_latest_file(path, fid):
    """Get the most recent file for a given task id."""
    import os

    files = os.listdir(path)
    files = [f for f in files if fid in f]
    files.sort()
    if files:
        return f"{path}/{files[-1]}"
    else:
        return None


def run_task(
    lm,
    task,
    cache_path,
    results_path,
    n=5,
    nclusters=0,
):
    route = get_latest_file(cache_path, task.id)
    logger.debug(route)
    if route is None:
        return None

    # Run pipeline on preloaded data
    with open(route, "r") as f:
        data = json.load(f)

    routes = asyncio.run(lm.run_single_task(task, data, nroutes=n))

    fname = os.path.join(results_path, f"{task.id}.json")
    with open(fname, "w") as f:
        json.dump(routes, f)
    return routes


def mae(gt, lm):
    if isinstance(gt[0], bool):
        gt = [10 if x else 1 for x in gt]
    return np.mean(np.abs(np.array(gt) - np.array(lm)))
