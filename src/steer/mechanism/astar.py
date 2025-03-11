"""Search with A*"""

import heapq
from typing import Any, List

import numpy as np
import requests
from pydantic import BaseModel
from rdkit import Chem, RDLogger

from steer.logger import setup_logger

RDLogger.DisableLog("rdApp.*")

logger = setup_logger()


class Node(BaseModel):
    parent: Any
    smiles: str
    f: float
    g: float
    h: float

    def __lt__(self, other):
        """Comparison function for the heapq."""
        if self.f != other.f:
            return self.f < other.f
        elif self.g != other.g:
            return self.g < other.g
        else:
            return self.smiles < other.smiles

    def reconstruct_path(self):
        path = [self.smiles]
        current = self
        while current.parent is not None:
            path.append(current.parent.smiles)
            current = current.parent
        path.reverse()
        return ">>".join(path)


class MechanismSearch(BaseModel):
    policy: Any
    iteration: int = -1
    closed_list: List[Node] = []
    open_list: List[Node] = []

    def search(self, src: str, dest: str):
        """Search."""
        src_node = Node(parent=None, smiles=src, f=0, g=0, h=0)
        heapq.heappush(self.open_list, src_node)

        while len(self.open_list) > 0:
            self.iteration += 1
            logger.debug(
                f"It. {self.iteration:<10} Nodes visited: {len(self.closed_list):<20} Total calls: {self.policy.total_calls}"
            )
            solved = self.step(src, dest)
            if solved:
                self.reset()
                return solved

    def step(self, src, dest):
        p = heapq.heappop(self.open_list)
        self.closed_list.append(p)

        moves = self.possible_moves(p)
        values = self.policy(f"{src}>>{dest}", [], moves)
        for move, value in zip(moves, values):
            if self.is_solution(f"{src}>>{dest}", move):
                path = p.reconstruct_path() + f">>{move}"
                return path

            else:
                g_new = p.g + 1.0
                h_new = 10 - value
                f_new = g_new + h_new
                this_node = Node(
                    parent=p, smiles=move, f=f_new, g=g_new, h=h_new
                )
                if (
                    this_node not in self.open_list
                ):  # or (open_list[open_list.index(this_node)].f > f_new):
                    heapq.heappush(self.open_list, this_node)

    def possible_moves(self, state: Node):
        """Get possible moves."""
        url = "http://liacpc17.epfl.ch:5001/legal_moves"
        req = {"smiles": state.smiles, "highlight_reactive_center": False}
        response = requests.post(url, json=req).json()["smiles_list"]
        return response

    def is_solution(self, rxn: str, candidate: str):
        """Check if the reaction is solved."""
        tgt_prod = Chem.MolFromSmiles(rxn.split(">>")[-1])
        curr_prod = Chem.MolFromSmiles(candidate)

        if Chem.MolToSmiles(tgt_prod) == Chem.MolToSmiles(curr_prod):
            return True
        return False

    def reset(self):
        self.iteration = -1
        self.closed_list = []
        self.open_list = []


class RandomPolicy(BaseModel):
    total_calls: int = 0

    def __call__(self, rxn: str, history: List[str], moves: List[str]):
        self.total_calls += len(moves)
        return np.random.choice(np.arange(10), len(moves))


class RuleBased(BaseModel):
    total_calls: int = 0

    def __call__(self, rxn: str, history: List[str], moves: List[str]):
        self.total_calls += len(moves)
        return [self.value_one_move(m) for m in moves]

    def value_one_move(self, move: str):
        # If any charge >1 -> return 0
        if "2-" in move:
            return 0
        if "3-" in move:
            return 0

        # If any [H-] -> return 5
        if "[H-]" in move:
            return 5

        # Else, return 10
        return 10


def format_path(path):
    return ">>".join(path)


def main():
    rp = RuleBased()
    ms = MechanismSearch(policy=rp)

    src = "CC=O.Cl"
    dest = "CC(Cl)O"
    sln = ms.search(src, dest)
    logger.info(sln)

    # for path in ms.search("CC=O.Cl>>CC(Cl)O"):
    #     logger.info(format_path(path))


if __name__ == "__main__":
    main()
