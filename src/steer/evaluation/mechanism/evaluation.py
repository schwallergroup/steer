"""Run evaluation script."""

import json
import os

import numpy as np
from tasks import load_default_tasks

from steer.logger import setup_logger

logger = setup_logger()

path = os.path.dirname(os.path.abspath(__file__))


def run_task(task, lm):
    """Run a task and return the results.
    Output is list of lists. list[0] is always correct one."""

    results = []
    for i, step in enumerate(task.steps[:-1]):
        possible_moves = [step, *task.step_options[i]]
        hist = task.steps[:i]
        scores = lm(task.rxn, hist, possible_moves)
        results.append(scores)
    return results


if __name__ == "__main__":
    tasks = load_default_tasks()
    scorer = lambda rxn, hist, moves: np.random.rand(len(moves)).tolist()
    logger.info(f"Loaded {len(tasks)} tasks.")

    for task in tasks:
        logger.info(f"Running task {task.id}")
        results = run_task(task, scorer)
        gt, lm = task.evaluate(results)
        corr = np.corrcoef(gt, lm)[0, 1]
        logger.debug(f"Correlation: {corr}")
