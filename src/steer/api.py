# -*- coding: utf-8 -*-

"""Main code."""

import asyncio
import json
import logging
import os
from datetime import datetime
from time import sleep
from typing import List

import click
import numpy as np
from aizynthfinder.analysis.routes import RouteCollection
from aizynthfinder.reactiontree import ReactionTree

# from steer.evaluation.sequential import get_latest_file, load_default_tasks
from steer.logger import setup_logger

logger = setup_logger(__name__)


async def run_amol(
    rxn: str, mechanisms: List[List[str]], lm  # List of mechanisms
):
    results = await asyncio.gather(
        *[lm.run(rxn, m, return_score=True) for m in mechanisms]
    )
    return results


def eval_path(rxn, seq, lm):
    smi_list = [f"{seq[i]}>>{seq[i+1]}" for i in range(len(seq) - 1)]
    list1 = [smi_list[: i + 1] for i in range(len(smi_list))]
    result = asyncio.run(run_amol(rxn, list1, lm))
    for i, r in enumerate(result):
        logger.debug(f"Path {i+1}: {r}, len: {len(list1[i])}")
    return result


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
