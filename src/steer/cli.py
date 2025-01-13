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

from steer.evaluation.synthesis import get_latest_file, load_default_tasks
from steer.logger import setup_logger

from .api import *

__all__ = [
    "main",
]

logger = setup_logger(__name__)


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


@mech.command()
def filter():
    """Run one example."""
    from steer.mechanism.sequential import LM

    prompt = "steer.mechanism.prompts.alphamol"
    lm = LM(
        prompt=prompt,
        model="claude-3-5-sonnet",
        project_name=f"steer-mech-{prompt}",
    )

    correct_path = [
        "C1CCCCC1=O.F",
        "[F-].[H+].[H][C]1([H])[C](=[O])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]1([H])[H]",
        "[F-].[H+].[H][C]1([H])[C+]([O-])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]1([H])[H]",
        "[H+].[H][C]1([H])[C]([F])([O-])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]1([H])[H]",
        "[H][C]1([H])[C]([F])([O][H])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]1([H])[H]",
    ]

    anti_chem_correct_path = [
        "C1CCCCC1=O.F",
        "[F+].[H-].[H][C]1([H])[C](=[O])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]1([H])[H]",
        "[F+].[H-].[H][C]1([H])[C-]([O+])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]1([H])[H]",
        "[H-].[H][C]1([H])[C]([F])([O+])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]1([H])[H]",
        "[H][C]1([H])[C]([F])([O][H])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]1([H])[H]",
    ]

    crap_path = [
        "C1CCCCC1=O.F",
        "F.[H][C]1([H])[C-]([O+])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]1([H])[H]",
        "[H][C]1([H])[C-]([O][FH+])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]1([H])[H]",
        "[H][C]1([H])[C-]([O][F+2])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]1([H])[H].[H-]",
        "[H][C]1([H])[C]2([O][F+]2)[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]1([H])[H].[H-]",
        "[H][C]1([H])[C]2([O+].[F]2)[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]1([H])[H].[H-]",
        "[H][C]1([H])[C]2([O][H].[F]2)[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]1([H])[H]",
    ]

    debatable_path = [
        "C1CCCCC1=O.F",
        "[H+].[F-].[H][C]1([H])[C](=[O])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]1([H])[H]",
        "[F-].[H][C]1([H])[C](=[O+][H])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]1([H])[H]",
        "[F-].[H][C]1([H])[C+]([O][H])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]1([H])[H]",
        "[H][C]1([H])[C](F)([O][H])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]1([H])[H]",
    ]

    rxn = f"C1CCCCC1=O.F>>[H][C]1([H])[C]([F])([O][H])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]1([H])[H]"
    logger.info("Correct path")
    eval_path(rxn, correct_path, lm)
    logger.info("Anti-chem path")
    eval_path(rxn, anti_chem_correct_path, lm)
    logger.info("Crap path")
    eval_path(rxn, crap_path, lm)
    logger.info("Debatable path")
    eval_path(rxn, debatable_path, lm)


@mech.command()
def sample():
    from steer.mechanism.sequential import LM

    logger.info("Running one example.")

    prompt = "steer.mechanism.prompts.alphamol_w_query"
    lm = LM(
        prompt=prompt,
        model="claude-3-5-sonnet",
        project_name=f"steer-mech-{prompt}",
    )

    rxn = f"CC(C)(O)c1cccc(-n2[nH]c(=O)c3cnc(Nc4ccc(Br)cc4)nc32)n1.C=CCBr>>C=CCn1c(=O)c2cnc(Nc3ccc(Br)cc3)nc2n1-c1cccc(C(C)(C)O)n1"

    options = [
        "[H+].[H][C]([H])=[C]([H])[C]([H])([H])[Br].[H][O][C]([C]1=[C]([H])[C]([H])=[C]([H])[C]([N]2[N-][C](=[O])[C]3=[C]([H])[N]=[C]([N]([H])[C]4=[C]([H])[C]([H])=[C]([Br])[C]([H])=[C]4[H])[N]=[C]32)=[N]1)([C]([H])([H])[H])[C]([H])([H])[H]",  # the good
        "[H-].[H][C]([H])=[C]([H])[C]([H])([H])[Br].[H][O][C]([C]1=[C]([H])[C]([H])=[C]([H])[C]([N]2[N+][C](=[O])[C]3=[C]([H])[N]=[C]([N]([H])[C]4=[C]([H])[C]([H])=[C]([Br])[C]([H])=[C]4[H])[N]=[C]32)=[N]1)([C]([H])([H])[H])[C]([H])([H])[H]",  # the anti
        "[H-].[H][C]([H])=[C]([H])[C]([H])([H])[Br].[H][O][C]([C]1=[C]([H])[C]([H])=[C]([H])[C]([N]2[C]3=[N][C]([N+][C]4=[C]([H])[C]([H])=[C]([Br])[C]([H])=[C]4[H])=[N][C]([H])=[C]3[C](=[O])[N]2[H])=[N]1)([C]([H])([H])[H])[C]([H])([H])[H]",
        "[H][C]([H])=[C]([H])[C]([H])([H])[Br].[H][C]1=[C+][N]=[C]([N]2[C]3=[N][C]([N]([H])[C]4=[C]([H])[C]([H])=[C]([Br])[C]([H])=[C]4[H])=[N][C]([H])=[C]3[C](=[O])[N]2[H])[C]([H])=[C]1[H].[H][O][C-]([C]([H])([H])[H])[C]([H])([H])[H]",
        "[H-].[H][C]([H])=[C]([H])[C]([H])([H])[Br].[H][C]1=[C]2[C](=[O])[N]([H])[N]([C]3=[N][C]([C]([O+])([C]([H])([H])[H])[C]([H])([H])[H])=[C]([H])[C]([H])=[C]3[H])[C]2=[N][C]([N]([H])[C]2=[C]([H])[C]([H])=[C]([Br])[C]([H])=[C]2[H])=[N]1",
    ]

    logger.info("Full rxn step")
    result = asyncio.run(
        run_amol(rxn, [[f"{rxn.split('>>')[0]}>>{o}"] for o in options], lm)
    )

    for i, r in enumerate(result):
        print(f"{options[i]} {r}")


@click.group()
@click.version_option()
@click.option("--model", default="gpt-4o", help="Model to use")
@click.option("--vision", default=False, help="Pass reactions as images")
@click.pass_context
def synth(ctx, model, vision):
    """CLI for steer."""
    import wandb
    from steer.llm.sequential import LM

    dt_name = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    CACHE_PATH = "data/fullroute_no_feasibility"
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
        },
    )

    lm = LM(
        prompt=prompt,
        model=model,
        vision=vision,
        project_name=project,
    )
    tasks = load_default_tasks()

    # Initialize ctx.obj if it doesn't exist
    if ctx.obj is None:
        ctx.obj = {}

    # Store objects in context
    ctx.obj["lm"] = lm
    ctx.obj["tasks"] = tasks
    ctx.obj["results_dir"] = RESULTS_DIR
    ctx.obj["cache_path"] = CACHE_PATH
    ctx.obj["wandb"] = wandb


@synth.command()
@click.option("--task", default="", help="Task id to run")
@click.pass_context
def one_task(ctx, task):
    """Run one example of synthesis re-ranking."""
    lm = ctx.obj["lm"]
    tasks = ctx.obj["tasks"]
    wandb = ctx.obj["wandb"]

    for i, t in enumerate(tasks):
        if t.id != task:
            continue

        routes = run_task(
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

        mae_val = mae(gt_score, lmscore)
        cor_val = np.corrcoef(gt_score, lmscore)[0, 1]
        wandb.log({f"mae_{t.id}": mae_val, f"corr_{t.id}": cor_val})


@synth.command()
@click.pass_context
def all_task(ctx):
    """Run all tasks in benchmark."""
    lm = ctx.obj["lm"]
    tasks = ctx.obj["tasks"]
    wandb = ctx.obj["wandb"]

    metrics = {
        "MAE": 0,
        "Corr": 0,
    }

    tasks = load_default_tasks()
    for i, task in enumerate(tasks):
        routes = run_task(
            lm,
            task,
            n=200,
            nclusters=0,
            cache_path=ctx.obj["cache_path"],
            results_path=ctx.obj["results_dir"],
        )
        if routes is None:
            continue

        # Evaluate
        gt_score, lmscore = task.evaluate(routes)

        mae_val = mae(gt_score, lmscore)
        cor_val = np.corrcoef(gt_score, lmscore)[0, 1]
        metrics["MAE"] += mae_val
        metrics["Corr"] += cor_val
        wandb.log({f"mae_{task.id}": mae_val, f"corr_{task.id}": cor_val})
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
