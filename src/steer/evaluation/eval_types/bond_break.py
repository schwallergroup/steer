
from .base import BaseScoring
import pandas as pd
from rdkit import Chem


class SpecificBondBreak(BaseScoring):
    def __init__(self, config):
        if config.get("bond_to_break") is None:
            raise ValueError("bond_to_break must be provided in config")
        else:
            self.atom_1 = config["bond_to_break"]["atom_1"]
            self.atom_2 = config["bond_to_break"]["atom_2"]

    def route_scoring(self, x):
        """Disconnection happens (!=-1), + should happen late-stage roughly."""
        if x == -1:
            return 99
        return x-1

    def hit_condition(self, d):
        """Determine if the bond between A1 and A2 is broken in current reaction."""

        rxn = d['metadata']['mapped_reaction_smiles'].split(">>")
        prod = Chem.MolFromSmiles(rxn[0])
        reacts = [Chem.MolFromSmiles(r) for r in rxn[1].split(".")]

        if (self.atom_1 in  [a.GetAtomMapNum() for a in prod.GetAtoms()]) and (self.atom_2 in  [a.GetAtomMapNum() for a in prod.GetAtoms()]):
            for r in reacts:
                if (self.atom_1 in [a.GetAtomMapNum() for a in r.GetAtoms()]) ^ (self.atom_2 in [a.GetAtomMapNum() for a in r.GetAtoms()]):
                    return True
        return False

    def __call__(self, data):
        return self.where_condition_met(data, self.route_scoring)



# class SpecificBondBreak_2(BaseScoring):
#     def hit_condition(self, d):
#         # Here, the question is when the piperidine rings become two separate molecules.
#         ring1 = 17 # C17
#         ring2 = 21 # N21

#         rxn = d['metadata']['mapped_reaction_smiles'].split(">>")
#         prod = Chem.MolFromSmiles(rxn[0])
#         reacts = [Chem.MolFromSmiles(r) for r in rxn[1].split(".")]

#         if (ring1 in  [a.GetAtomMapNum() for a in prod.GetAtoms()]) and (ring2 in  [a.GetAtomMapNum() for a in prod.GetAtoms()]):
#             for r in reacts:
#                 if (ring1 in [a.GetAtomMapNum() for a in r.GetAtoms()]) ^ (ring2 in [a.GetAtomMapNum() for a in r.GetAtoms()]):
#                     return True
#         return False



# class SpecificBondBreak_3(BaseScoring):

#     def hit_condition(self, d):
#         # Here, the question is when the piperidine rings become two separate molecules.
#         ring1 = 24 # N24
#         ring2 = 26 # C26

#         rxn = d['metadata']['mapped_reaction_smiles'].split(">>")
#         prod = Chem.MolFromSmiles(rxn[0])
#         reacts = [Chem.MolFromSmiles(r) for r in rxn[1].split(".")]

#         if (ring1 in  [a.GetAtomMapNum() for a in prod.GetAtoms()]) and (ring2 in  [a.GetAtomMapNum() for a in prod.GetAtoms()]):
#             for r in reacts:
#                 if (ring1 in [a.GetAtomMapNum() for a in r.GetAtoms()]) ^ (ring2 in [a.GetAtomMapNum() for a in r.GetAtoms()]):
#                     return True
#         return False


# class SpecificBondBreak_4(BaseScoring):

#     def hit_condition(self, d):
#         # Here, the question is when the piperidine rings become two separate molecules.
#         ring1 = 36 # C36
#         ring2 = 38 # C38

#         rxn = d['metadata']['mapped_reaction_smiles'].split(">>")
#         prod = Chem.MolFromSmiles(rxn[0])
#         reacts = [Chem.MolFromSmiles(r) for r in rxn[1].split(".")]

#         if (ring1 in  [a.GetAtomMapNum() for a in prod.GetAtoms()]) and (ring2 in  [a.GetAtomMapNum() for a in prod.GetAtoms()]):
#             for r in reacts:
#                 if (ring1 in [a.GetAtomMapNum() for a in r.GetAtoms()]) ^ (ring2 in [a.GetAtomMapNum() for a in r.GetAtoms()]):
#                     return True
#         return False