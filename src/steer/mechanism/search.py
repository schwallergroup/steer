"""Search algorithm for finding mechanism."""

from typing import Any, List

import numpy as np
import requests
from pydantic import BaseModel
from rdkit import Chem, RDLogger

from steer.logger import setup_logger
from steer.mechanism.molecule_set import legal_moves_from_smiles

RDLogger.DisableLog("rdApp.*")

logger = setup_logger()


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class MechanismSearch(BaseModel):
    policy: Any
    max_steps: int = 4
    beam_size: int = 3

    def search(self, rxn: str):
        """Search for possible mechanisms."""

        solutions = []
        paths = [[rxn.split(">>")[0]]]
        for step in range(self.max_steps):
            logger.debug(f"Step {step}. Paths: {len(paths)}")
            new_paths = []
            for path in paths:
                nps, vls = self.step(rxn, path)
                new_paths.extend(nps)
            paths = new_paths

            # Check if solved
            for path in paths:
                if self.is_solution(rxn, path):
                    logger.info(f"Found solution at iter {step}")
                    solutions.append(path)
                    yield path

        if not solutions:
            logger.info("No solution found.")

    def step(self, rxn: str, history: List[str]):
        """Take the next step for one trajectory."""

        # Get possible moves
        state = history[-1]
        moves = self.possible_moves(state)

        # Estimate value of each move
        values = self.policy(rxn, history, moves)
        ps = softmax(values)

        # Sample some moves, return index
        idx = np.random.choice(
            np.arange(len(moves)),
            min(self.beam_size, len(ps)),
            replace=False,
            p=ps,
        )
        vals = [values[i] for i in idx]
        mvs = [moves[i] for i in idx]
        new_paths = [[*history, m] for m in mvs]
        return new_paths, vals

    def is_solution(self, rxn: str, candidate: List[str]):
        """Check if the reaction is solved."""
        tgt_prod = Chem.MolFromSmiles(rxn.split(">>")[-1])
        curr_prod = Chem.MolFromSmiles(candidate[-1])

        if Chem.MolToSmiles(tgt_prod) == Chem.MolToSmiles(curr_prod):
            return True
        return False

    def possible_moves(self, state: str):
        """Get possible moves."""

        response = legal_moves_from_smiles(
            state, highlight_reactive_center=False
        )["smiles_list"]

        try:
            return np.random.choice(
                response, min(self.beam_size, len(response)), replace=False
            ).tolist()
        except:
            return response


class RandomPolicy(BaseModel):
    def __call__(self, rxn: str, history: List[str], moves: List[str]):
        return np.random.choice(np.arange(10), len(moves))


def format_path(path):
    return ">>".join(path)


if __name__ == "__main__":
    rp = RandomPolicy()
    ms = MechanismSearch(policy=rp, max_steps=4, beam_size=10)
    for path in ms.search("CC=O.Cl>>CC(Cl)O"):
        logger.info(format_path(path))
