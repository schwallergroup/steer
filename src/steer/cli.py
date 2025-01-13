# -*- coding: utf-8 -*-

import asyncio
import logging
import os
from datetime import datetime
from time import sleep
from typing import List

import click
import numpy as np

from steer.logger import setup_logger

__all__ = [
    "main",
]

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
def filter():
    """Run one example."""
    from steer.llm.fullroute import LM

    lm = LM(
        prompt="steer.llm.prompts.alphamol",
        model="claude-3-5-sonnet",
        project_name="steer-mechanism-test",
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


@main.command()
def alphamol():
    from steer.llm.fullroute import LM

    logger.info("Running one example.")

    prompt = "steer.llm.prompts.alphamol_w_query"
    lm = LM(
        prompt=prompt,
        model="claude-3-5-sonnet",
        # project_name=f"steer-{prompt}",
    )

    rxn = f"CC(C)(O)c1cccc(-n2[nH]c(=O)c3cnc(Nc4ccc(Br)cc4)nc32)n1.C=CCBr>>C=CCn1c(=O)c2cnc(Nc3ccc(Br)cc3)nc2n1-c1cccc(C(C)(C)O)n1"

    with open("all_next_smiles_1.txt", "r") as f:
        options = eval(f.read())

    # options = [
    #     "[H+].[H][C]([H])=[C]([H])[C]([H])([H])[Br].[H][O][C]([C]1=[C]([H])[C]([H])=[C]([H])[C]([N]2[N-][C](=[O])[C]3=[C]([H])[N]=[C]([N]([H])[C]4=[C]([H])[C]([H])=[C]([Br])[C]([H])=[C]4[H])[N]=[C]32)=[N]1)([C]([H])([H])[H])[C]([H])([H])[H]", # the good
    #     "[H-].[H][C]([H])=[C]([H])[C]([H])([H])[Br].[H][O][C]([C]1=[C]([H])[C]([H])=[C]([H])[C]([N]2[N+][C](=[O])[C]3=[C]([H])[N]=[C]([N]([H])[C]4=[C]([H])[C]([H])=[C]([Br])[C]([H])=[C]4[H])[N]=[C]32)=[N]1)([C]([H])([H])[H])[C]([H])([H])[H]", # the anti
    #     "[H-].[H][C]([H])=[C]([H])[C]([H])([H])[Br].[H][O][C]([C]1=[C]([H])[C]([H])=[C]([H])[C]([N]2[C]3=[N][C]([N+][C]4=[C]([H])[C]([H])=[C]([Br])[C]([H])=[C]4[H])=[N][C]([H])=[C]3[C](=[O])[N]2[H])=[N]1)([C]([H])([H])[H])[C]([H])([H])[H]",
    #     "[H][C]([H])=[C]([H])[C]([H])([H])[Br].[H][C]1=[C+][N]=[C]([N]2[C]3=[N][C]([N]([H])[C]4=[C]([H])[C]([H])=[C]([Br])[C]([H])=[C]4[H])=[N][C]([H])=[C]3[C](=[O])[N]2[H])[C]([H])=[C]1[H].[H][O][C-]([C]([H])([H])[H])[C]([H])([H])[H]",
    #     "[H-].[H][C]([H])=[C]([H])[C]([H])([H])[Br].[H][C]1=[C]2[C](=[O])[N]([H])[N]([C]3=[N][C]([C]([O+])([C]([H])([H])[H])[C]([H])([H])[H])=[C]([H])[C]([H])=[C]3[H])[C]2=[N][C]([N]([H])[C]2=[C]([H])[C]([H])=[C]([Br])[C]([H])=[C]2[H])=[N]1",
    # ]

    logger.info("Full rxn step")
    result = asyncio.run(
        run_amol(rxn, [[f"{rxn.split('>>')[0]}>>{o}"] for o in options], lm)
    )

    for i, r in enumerate(result):
        print(f"{options[i]} {r}")

    # options = [
    #     "[H+].[H][C]([H])=[C]([H])[C]([H])([H])[Br].[H][O][C]([C]1=[C]([H])[C]([H])=[C]([H])[C]([N]2[N-][C](=[O])[C]3=[C]([H])[N]=[C]([N]([H])[C]4=[C]([H])[C]([H])=[C]([Br])[C]([H])=[C]4[H])[N]=[C]32)=[N]1)([C]([H])([H])[H])[C]([H])([H])[H]", # the good
    #     "[H-].[H][C]([H])=[C]([H])[C]([H])([H])[Br].[H][O][C]([C]1=[C]([H])[C]([H])=[C]([H])[C]([N]2[N+][C](=[O])[C]3=[C]([H])[N]=[C]([N]([H])[C]4=[C]([H])[C]([H])=[C]([Br])[C]([H])=[C]4[H])[N]=[C]32)=[N]1)([C]([H])([H])[H])[C]([H])([H])[H]", # the anti
    #     "[H-].[H][C]([H])=[C]([H])[C]([H])([H])[Br].[H][O][C]([C]1=[C]([H])[C]([H])=[C]([H])[C]([N]2[C]3=[N][C]([N+][C]4=[C]([H])[C]([H])=[C]([Br])[C]([H])=[C]4[H])=[N][C]([H])=[C]3[C](=[O])[N]2[H])=[N]1)([C]([H])([H])[H])[C]([H])([H])[H]",
    #     "[H][C]([H])=[C]([H])[C]([H])([H])[Br].[H][C]1=[C+][N]=[C]([N]2[C]3=[N][C]([N]([H])[C]4=[C]([H])[C]([H])=[C]([Br])[C]([H])=[C]4[H])=[N][C]([H])=[C]3[C](=[O])[N]2[H])[C]([H])=[C]1[H].[H][O][C-]([C]([H])([H])[H])[C]([H])([H])[H]",
    #     "[H-].[H][C]([H])=[C]([H])[C]([H])([H])[Br].[H][C]1=[C]2[C](=[O])[N]([H])[N]([C]3=[N][C]([C]([O+])([C]([H])([H])[H])[C]([H])([H])[H])=[C]([H])[C]([H])=[C]3[H])[C]2=[N][C]([N]([H])[C]2=[C]([H])[C]([H])=[C]([Br])[C]([H])=[C]2[H])=[N]1",
    # ]

    # logger.info("Single prods")
    # result = asyncio.run(run_amol(rxn, [[f"{rxn.split('>>')[0]}>>{o}"] for o in options], lm))

    # for i,r in enumerate(result):
    #     print(f"{options[i]} {r}")


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

        if task.id in [
            "e579d80f176371344bab95ea15e6b9ab",
            "4bfe366ec7f5d64678d500f9084cbb35",
            "dfc8116ec63329c437281f7a40dda876",
        ]:
            print(task.id)
            print(gt_score)
            print(lmscore)
            print("----")

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
