# -*- coding: utf-8 -*-

import asyncio
import logging
import os
from datetime import datetime
from time import sleep

import click
import numpy as np

from steer.logger import setup_logger

__all__ = [
    "main",
]

logger = setup_logger(__name__)

import json
from typing import List

from aizynthfinder.analysis.routes import RouteCollection
from aizynthfinder.reactiontree import ReactionTree

from steer.evaluation.sequential import get_latest_file, load_default_tasks

dt_name = datetime.now().strftime("%Y-%m-%d_%H%M%S")
PREMADE_PATH = "data/fullroute_no_feasibility"
RESULTS_DIR = f"data/{dt_name}"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Add route clustering


# Load some sample routes
def cluster_routes(data: List[dict], nclusters=10):
    routes = [ReactionTree.from_dict(d) for d in data]
    rc = RouteCollection(routes)
    index_map = rc.cluster(
        n_clusters=nclusters,
        distances_model="lstm",
        model_path="data/models/chembl_10k_route_distance_model.ckpt",
    )
    rts = [rc.clusters[i][0] for i in range(nclusters)]
    return rts, index_map


def run_task(
    lm,
    task,
    cache_path=PREMADE_PATH,
    results_path=RESULTS_DIR,
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

    if nclusters:
        rts, index_map = cluster_routes(data, nclusters=nclusters)
        routes = [rt["reaction_tree"].to_dict() for rt in rts]
        routes = asyncio.run(
            lm.run_single_task(task, routes, nroutes=nclusters)
        )

        # Map back to original indices
        for i, r in enumerate(data):
            data[i]["lmdata"] = routes[index_map[i]]["lmdata"]
        routes = data

    else:
        routes = asyncio.run(lm.run_single_task(task, data, nroutes=n))

    fname = os.path.join(results_path, f"{task.id}.json")
    with open(fname, "w") as f:
        json.dump(routes, f)
    return routes


def mae(gt, lm):
    if isinstance(gt[0], bool):
        gt = [10 if x else 1 for x in gt]
    return np.mean(np.abs(np.array(gt) - np.array(lm)))


@click.group()
@click.version_option()
def main():
    """CLI for steer."""


@main.command()
def run():
    from steer.llm.sequential import main

    asyncio.run(main())


@main.command()
@click.option("--model", default="gpt-4o", help="Model to use")
@click.option("--vision", default=False, help="Pass reactions as images")
@click.option("--ncluster", default=0, help="Cluster routes")
def all_tasks(model, vision, ncluster):

    import wandb
    from steer.llm.sequential import LM

    prompt = "steer.llm.prompts.route_opt"
    wandb.init(
        project="steer-test",
        config={
            "model": model,
            "vision": vision,
            "ncluster": ncluster,
            "prompt": prompt,
        },
    )

    lm = LM(
        prompt=prompt,
        model=model,
        vision=vision,
        # project_name="steer-test",
    )

    metrics = {
        "MAE": 0,
        "Corr": 0,
    }
    tasks = load_default_tasks()
    for i, task in enumerate(tasks):
        if task.eval_type not in [
            "RingBreakDepth",
            "MultiRxnCond",
            "SpecificBondBreak",
        ]:
            continue
        sleep(2)
        routes = run_task(lm, task, n=200, nclusters=ncluster)
        if routes is None:
            continue

        # Evaluate
        gt_score, lmscore = task.evaluate(routes)

        if task.id in ["e579d80f176371344bab95ea15e6b9ab", "4bfe366ec7f5d64678d500f9084cbb35", "dfc8116ec63329c437281f7a40dda876"]:
            print(task.id)
            print(gt_score)
            print(lmscore)
            print('----')

        mae_val = mae(gt_score, lmscore)
        cor_val = np.corrcoef(gt_score, lmscore)[0, 1]
        metrics["MAE"] += mae_val
        metrics["Corr"] += cor_val
        wandb.log({f"mae_{task.id}": mae_val, f"corr_{task.id}": cor_val})

    wandb.log(
        {
            "mean_mae": metrics["MAE"] / len(tasks),
            "mean_corr": metrics["Corr"] / len(tasks),
        }
    )

    wandb.finish()


if __name__ == "__main__":
    main()
