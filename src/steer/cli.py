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

@click.group()
@click.version_option()
def main():
    """CLI for steer."""

@main.command()
def alphamol():
    """Run one example."""
    from steer.llm.fullroute import LM
    logger.info("Running one example.")

    lm = LM(
        prompt="steer.llm.prompts.alphamol",
        model="claude-3-5-sonnet",
        project_name="steer-mechanism-test",
    )

    start_smiles = "C1CCCCC1=O.F"

    list_smiles = [ # All legal ionization + attacks from cyclohexanone + HF
        '[H][F].[H][C+]([H])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C](=[O])[C-]([H])[H]',
        '[H][F].[H][C+]([H])[C](=[O])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C-]([H])[H]',
        '[H][F].[H][C+]([H])[C]([H])([H])[C]([H])([H])[C](=[O])[C]([H])([H])[C-]([H])[H]',
        '[H][F].[H][C+]([H])[C]([H])([H])[C](=[O])[C]([H])([H])[C]([H])([H])[C-]([H])[H]',
        '[H][F].[H][C-]([H])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C+]=[O]',
        '[H][F].[H][C+]([H])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C-]=[O]',
        '[H][C]1([H])[C-]([O+])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]1([H])[H].[H][F]',
        '[H][C]1([H])[C+]([O-])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]1([H])[H].[H][F]',
        '[H+].[H][F].[H][C-]1[C](=[O])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]1([H])[H]',
        '[H-].[H][F].[H][C+]1[C](=[O])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]1([H])[H]',
        '[H+].[H][F].[H][C-]1[C]([H])([H])[C](=[O])[C]([H])([H])[C]([H])([H])[C]1([H])[H]',
        '[H-].[H][F].[H][C+]1[C]([H])([H])[C](=[O])[C]([H])([H])[C]([H])([H])[C]1([H])[H]',
        '[H+].[H][F].[H][C-]1[C]([H])([H])[C]([H])([H])[C](=[O])[C]([H])([H])[C]1([H])[H]',
        '[H-].[H][F].[H][C+]1[C]([H])([H])[C]([H])([H])[C](=[O])[C]([H])([H])[C]1([H])[H]',
        '[F-].[H+].[H][C]1([H])[C](=[O])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]1([H])[H]',
        '[F+].[H-].[H][C]1([H])[C](=[O])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]1([H])[H]',
    ]

    correct_path = [
        "C1CCCCC1=O.F",
        '[F-].[H+].[H][C]1([H])[C](=[O])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]1([H])[H]',
        '[F-].[H+].[H][C]1([H])[C+]([O-])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]1([H])[H]',
        '[H+].[H][C]1([H])[C]([F])([O-])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]1([H])[H]',
        '[H][C]1([H])[C]([F])([O][H])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]1([H])[H]',
    ]

    rxn = f"{start_smiles}>>[H][C]1([H])[C]([F])([O][H])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]1([H])[H]"
    smi_list = [f"{correct_path[i]}>>{correct_path[i+1]}" for i in range(len(correct_path)-1)]

    result = asyncio.run(run_amol(rxn, [smi_list], lm))

    print(result)
    return result


if __name__ == "__main__":
    main()
