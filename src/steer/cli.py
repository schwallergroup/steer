# -*- coding: utf-8 -*-

import asyncio
import logging

import click
from steer.logger import setup_logger
from typing import List

__all__ = [
    "main",
]

logger = setup_logger(__name__)

async def run_amol(
    rxn: str,
    mechanisms: List[List[str]], # List of mechanisms
    lm
):
    results = await asyncio.gather(*[lm.run(rxn, m, return_score=True) for m in mechanisms])
    return results


def eval_path(rxn, seq, lm):
    smi_list = [f"{seq[i]}>>{seq[i+1]}" for i in range(len(seq)-1)]
    list1 = [smi_list[:i+1] for i in range(len(smi_list))]
    result = asyncio.run(run_amol(rxn, list1, lm))
    for i,r in enumerate(result):
        logger.debug(f"Path {i+1}: {r}, len: {len(list1[i])}")
    return result


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
        '[F-].[H+].[H][C]1([H])[C](=[O])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]1([H])[H]',
        '[F-].[H+].[H][C]1([H])[C+]([O-])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]1([H])[H]',
        '[H+].[H][C]1([H])[C]([F])([O-])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]1([H])[H]',
        '[H][C]1([H])[C]([F])([O][H])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]1([H])[H]',
    ]
    
    anti_chem_correct_path = [
        "C1CCCCC1=O.F",
        '[F+].[H-].[H][C]1([H])[C](=[O])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]1([H])[H]',
        '[F+].[H-].[H][C]1([H])[C-]([O+])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]1([H])[H]',
        '[H-].[H][C]1([H])[C]([F])([O+])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]1([H])[H]',
        '[H][C]1([H])[C]([F])([O][H])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]1([H])[H]',
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

    lm = LM(
        prompt="steer.llm.prompts.alphamol2",
        model="claude-3-5-sonnet",
        project_name="steer-mechanism-am",
    )

    rxn = f"CC(C)(O)c1cccc(-n2[nH]c(=O)c3cnc(Nc4ccc(Br)cc4)nc32)n1.C=CCBr>>C=CCn1c(=O)c2cnc(Nc3ccc(Br)cc3)nc2n1-c1cccc(C(C)(C)O)n1"

    with open("all_next_smiles_1.txt", "r") as f:
        options = eval(f.read())

    result = asyncio.run(run_amol(rxn, [[f"{rxn.split('>>')[0]}>>{o}"] for o in options], lm))

    for i,r in enumerate(result):
        print(f"{options[i]} {r}")

    logger.info("Correct path")



if __name__ == "__main__":
    main()
