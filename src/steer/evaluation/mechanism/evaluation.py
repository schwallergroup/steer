"""Run evaluation script."""

import asyncio
import json
import os
from typing import List

import numpy as np

import wandb
from steer.logger import setup_logger

from .tasks import load_default_tasks

logger = setup_logger(__name__)

path = os.path.dirname(os.path.abspath(__file__))

from steer.mechanism.molecule_set import MoleculeSet


def make_rxns(steps: List[str]):
    if len(steps) < 2:
        return []
    # return [f"{steps[i]}>>{steps[i+1]}" for i in range(len(steps) - 1)]
    return [
        f"{MoleculeSet(steps[i]).rdkit_canonical_smiles}>>{MoleculeSet(steps[i+1]).rdkit_canonical_smiles}"
        for i in range(len(steps) - 1)
    ]


async def run_task(task, lm):
    """Run a task and return the results.
    Output is list of lists. list[0] is always correct one."""

    results = []
    for i, step in enumerate(task.steps[:-1]):
        possible_moves = [task.steps[i + 1], *task.step_options[i]]
        hist = make_rxns(task.steps[: i + 1])
        response = await asyncio.gather(
            *[
                lm.run(
                    rxn=task.rxn,
                    history=hist,
                    step=make_rxns([task.steps[i], move])[0],
                    expert_description=task.expert_description,
                )
                for move in possible_moves
            ]
        )
        scores = [lm._parse_score(result) for result in response]
        results.append(scores)
    return results


async def main(
    prompt="steer.mechanism.prompts.preprint_prompt_last_step_plus_game",
    model="claude-3-5-sonnet",
    vision=False,
    project_name="steer-mechanism-test",
    tasks_user=None,
    expert_needed=False,
):
    from steer.mechanism.sequential import LM

    scorer = LM(
        prompt=prompt,
        model=model,
        vision=vision,
        project_name=project_name,
        prompt_needs_expert_description=expert_needed,
    )

    all_tasks = load_default_tasks()

    if tasks_user is None:
        tasks = all_tasks
    else:
        if isinstance(tasks_user, str):
            tasks_user = [tasks_user]
        tasks = [task for task in all_tasks if task.id in tasks_user]

    logger.info(f"Loaded {len(tasks)} tasks.")

    for task in tasks:
        logger.info(f"Running task {task.id}")
        results = await run_task(task, scorer)
        gt, lm = task.evaluate(results)
        corr = np.corrcoef(gt, lm)[0, 1]
        logger.debug(f"Correlation: {corr}")
        wandb.log(
            {
                f"task_{task.id}": results,
                f"gt_steps_{task.id}": task.steps,
                f"wrong_steps_{task.id}": task.step_options,
            }
        )


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
