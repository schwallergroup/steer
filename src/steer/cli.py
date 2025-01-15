# -*- coding: utf-8 -*-

import asyncio
import json
import logging
import os
from datetime import datetime
from time import sleep
from typing import List

import click
import numpy as np

from steer.evaluation import synthesis
from steer.logger import setup_logger

__all__ = [
    "main",
]

logger = setup_logger(__name__)


# Mechanisms


@click.group()
@click.version_option()
@click.option("--model", default="gpt-4o", help="Model to use")
@click.option("--vision", default=False, help="Pass reactions as images")
@click.pass_context
def mech(ctx, model, vision):
    """CLI for steer."""
    if ctx.obj is None:
        ctx.obj = {}

    ctx.obj["model"] = model
    ctx.obj["vision"] = vision


@mech.command()
@click.pass_context
def bench(ctx):
    """Run benchmar."""
    import wandb
    from steer.evaluation.mechanism.evaluation import main

    prompt = "steer.mechanism.prompts.alphamol_partial"
    project = "steer-mechbench"
    model = ctx.obj["model"]
    vision = ctx.obj["vision"]

    wandb.init(
        project=project,
        config={
            "model": model,
            "vision": vision,
            "prompt": prompt,
        },
    )

    asyncio.run(
        main(
            prompt=prompt,
            model=model,
            project_name=project,
        )
    )


# Synthesis


@click.group()
@click.version_option()
@click.option("--model", default="gpt-4o", help="Model to use")
@click.option("--vision", default=False, help="Pass reactions as images")
@click.pass_context
def synth(ctx, model, vision):
    """CLI for steer."""
    import wandb
    from steer.evaluation.synthesis import load_default_tasks
    from steer.llm.sequential import LM

    dt_name = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    CACHE_PATH = "data/synth_bench"
    RESULTS_DIR = f"data/{dt_name}"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    prompt = "steer.llm.prompts.route_opt"
    project = "steer-test"
    wandb.init(
        project=project,
        config={
            "model": model,
            "vision": vision,
            "prompt": prompt,
            "results_dir": RESULTS_DIR,
        },
    )

    lm = LM(
        prompt=prompt,
        model=model,
        vision=vision,
        project_name=project,
    )
    tasks = load_default_tasks()

    if ctx.obj is None:
        ctx.obj = {}

    ctx.obj["lm"] = lm
    ctx.obj["tasks"] = tasks
    ctx.obj["results_dir"] = RESULTS_DIR
    ctx.obj["cache_path"] = CACHE_PATH
    ctx.obj["wandb"] = wandb


@synth.command()
@click.pass_context
@click.option("--task", default=None, help="Task id to run")
def bench(ctx, task):
    """Run all tasks in benchmark."""
    lm = ctx.obj["lm"]
    tasks = ctx.obj["tasks"]
    wandb = ctx.obj["wandb"]

    metrics = {
        "MAE": 0,
        "Corr": 0,
    }

    for i, t in enumerate(tasks):
        if task is not None and t.id != task:
            continue

        routes = synthesis.run_task(
            lm,
            t,
            n=200,
            nclusters=0,
            cache_path=ctx.obj["cache_path"],
            results_path=ctx.obj["results_dir"],
        )
        if routes is None:
            continue

        # Evaluate
        gt_score, lmscore = t.evaluate(routes)

        mae_val = synthesis.mae(gt_score, lmscore)
        cor_val = np.corrcoef(gt_score, lmscore)[0, 1]
        metrics["MAE"] += mae_val
        metrics["Corr"] += cor_val
        wandb.log({f"mae_{t.id}": mae_val, f"corr_{t.id}": cor_val})
        sleep(2)

    wandb.log(
        {
            "mean_mae": metrics["MAE"] / len(tasks),
            "mean_corr": metrics["Corr"] / len(tasks),
        }
    )
    wandb.finish()


@click.group()
def main():
    """Steer CLI with multiple subcommands."""
    pass


main.add_command(synth)
main.add_command(mech)


if __name__ == "__main__":
    main()
