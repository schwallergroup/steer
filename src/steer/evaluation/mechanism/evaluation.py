"""Run evaluation script."""

import json
import os

import numpy as np
from tasks import load_default_tasks

from steer.logger import setup_logger

logger = setup_logger(__name__)

path = os.path.dirname(os.path.abspath(__file__))


async def run_task(task, lm):
    """Run a task and return the results.
    Output is list of lists. list[0] is always correct one."""

    results = []
    for i, step in enumerate(task.steps[:-1]):
        possible_moves = [step, *task.step_options[i]]
        hist = task.steps[:i]
        response = await asyncio.gather(*[
            lm.run(rxn=task.rxn, history=hist, step=move)
            for move in possible_moves
        ])
        scores = [lm._parse_score(result) for result in response]
        results.append(scores)
    return results


async def main():
    from steer.mechanism.sequential import LM
    scorer = LM(
        prompt="steer.mechanism.prompts.alphamol_partial",
        model="claude-3-5-sonnet",
        project_name="steer-mechanism-test",
    )

    tasks = load_default_tasks()
    logger.info(f"Loaded {len(tasks)} tasks.")

    for task in tasks:
        logger.info(f"Running task {task.id}")
        results = await run_task(task, scorer)
        gt, lm = task.evaluate(results)
        corr = np.corrcoef(gt, lm)[0, 1]
        logger.debug(f"Correlation: {corr}")



if __name__ == "__main__":
    import asyncio
    asyncio.run(main())