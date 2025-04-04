"""Scoring function for breaking a specific bond in a reaction."""

from rdkit import Chem

from steer.logger import setup_logger

from .base import BaseScoring

logger = setup_logger(__name__)


class SpecificBondBreak(BaseScoring):
    def __init__(self, config):
        if config.get("bond_to_break") is None:
            raise ValueError("bond_to_break must be provided in config")
        else:
            self.atom_1 = config["bond_to_break"]["atom_1"]
            self.atom_2 = config["bond_to_break"]["atom_2"]

    def route_scoring(self, x):
        """Disconnection happens (!=-1), + should happen late-stage roughly."""
        if x < 0:
            return 0  # Worst case - disconnection doesn't happen
        else:
            return (
                1 - x
            )  # Disconnection happens late-stage. The smaller x, the better.

    def hit_condition(self, d):
        """Determine if the bond between A1 and A2 is broken in current reaction."""

        rxn = d["metadata"]["mapped_reaction_smiles"].split(">>")
        prod = Chem.MolFromSmiles(rxn[0])
        reacts = [Chem.MolFromSmiles(r) for r in rxn[1].split(".")]

        if (self.atom_1 in [a.GetAtomMapNum() for a in prod.GetAtoms()]) and (
            self.atom_2 in [a.GetAtomMapNum() for a in prod.GetAtoms()]
        ):
            for r in reacts:
                if (
                    self.atom_1 in [a.GetAtomMapNum() for a in r.GetAtoms()]
                ) ^ (self.atom_2 in [a.GetAtomMapNum() for a in r.GetAtoms()]):
                    return True
        return False


if __name__ == "__main__":
    import json

    with open(
        "../../../data/2025-01-14_193014/4bfe366ec7f5d64678d500f9084cbb35.json",
        "r",
    ) as f:
        data = json.load(f)

    bs = SpecificBondBreak(
        config={"bond_to_break": {"atom_1": 29, "atom_2": 33}}
    )
    a, b = bs(data)
    for i, l in enumerate(a):
        if i == 7:
            print(l, b[i])
