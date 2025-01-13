"""Run evaluation script."""

# Benchmark stats
import json
import os

import numpy as np

from steer.logger import setup_logger

from .eval_types import *
from .tasks import load_default_tasks

logger = setup_logger()

path = os.path.dirname(os.path.abspath(__file__))

RESULTS_PATH = os.path.join(path, "../../../../../synthegy/steer/")
PROMPT_TYPE = "fullroute_no_feasibility"


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


def run_task(task):
    """Run a task and return the results."""
    filename = get_latest_file(f"{RESULTS_PATH}/{PROMPT_TYPE}", task.id)
    if filename is None:
        logger.debug(f"File not found for {task.id}")
        return None
    with open(filename, "r") as f:
        data = json.load(f)

    gt_score, lmscore = task.evaluate(data)
    return gt_score, lmscore


def metric(gt, lm):
    """MAE"""
    if isinstance(gt[0], bool):
        gt = [10 if x else 1 for x in gt]
    return np.mean(np.abs(np.array(gt) - np.array(lm)))


if __name__ == "__main__":
    tasks = load_default_tasks(path)
    mean_metric = 0
    for task in tasks:
        result = run_task(task)
        if result is not None:
            # logger.info(f"{task.id}: {result[0]}, {result[1]}")
            metric_val = metric(result[0], result[1])
            logger.debug(f"Metric: {metric_val}")
            mean_metric += metric_val
    logger.info(f"Mean Metric: {mean_metric / len(tasks)}")
