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


# Load some sample routes
# def cluster_routes(data: List[dict], nclusters=10):
#     from synthegy.analysis.routes import RouteCollection
#     from synthegy.reactiontree import ReactionTree

#     routes = [ReactionTree.from_dict(d) for d in data]
#     rc = RouteCollection(routes)
#     index_map = rc.cluster(
#         n_clusters=nclusters,
#         distances_model="lstm",
#         model_path="data/models/chembl_10k_route_distance_model.ckpt",
#     )
#     rts = [rc.clusters[i][0] for i in range(nclusters)]
#     return rts, index_map


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

    # if nclusters:
    #     rts, index_map = cluster_routes(data, nclusters=nclusters)
    #     routes = [rt["reaction_tree"].to_dict() for rt in rts]
    #     routes = asyncio.run(
    #         lm.run_single_task(task, routes, nroutes=nclusters)
    #     )

    #     # Map back to original indices
    #     for i, r in enumerate(data):
    #         data[i]["lmdata"] = routes[index_map[i]]["lmdata"]
    #     routes = data

    # else:
    routes = asyncio.run(lm.run_single_task(task, data, nroutes=n))

    fname = os.path.join(results_path, f"{task.id}.json")
    with open(fname, "w") as f:
        json.dump(routes, f)
    return routes


def mae(gt, lm):
    if isinstance(gt[0], bool):
        gt = [10 if x else 1 for x in gt]
    return np.mean(np.abs(np.array(gt) - np.array(lm)))


if __name__ == "__main__":
    pass
    # tasks = load_default_tasks(path)
    # mean_metric = 0
    # for task in tasks:
    #     result = run_task(task)
    #     if result is not None:
    #         # logger.info(f"{task.id}: {result[0]}, {result[1]}")
    #         metric_val = mae(result[0], result[1])
    #         logger.debug(f"Metric: {metric_val}")
    #         mean_metric += metric_val
    # logger.info(f"Mean Metric: {mean_metric / len(tasks)}")
