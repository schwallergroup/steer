
from .base import BaseScoring
import pandas as pd
from rdkit import Chem

class SpecificBondBreak_1(BaseScoring):
    def hit_condition(self, d):
        # Here, the question is when the piperidine rings become two separate molecules.
        ring1 = 29 # N29
        ring2 = 33 # N33

        rxn = d['metadata']['mapped_reaction_smiles'].split(">>")
        prod = Chem.MolFromSmiles(rxn[0])
        reacts = [Chem.MolFromSmiles(r) for r in rxn[1].split(".")]

        if (ring1 in  [a.GetAtomMapNum() for a in prod.GetAtoms()]) and (ring2 in  [a.GetAtomMapNum() for a in prod.GetAtoms()]):
            for r in reacts:
                if (ring1 in [a.GetAtomMapNum() for a in r.GetAtoms()]) ^ (ring2 in [a.GetAtomMapNum() for a in r.GetAtoms()]):
                    return True
        return False


class SpecificBondBreak_2(BaseScoring):
    def hit_condition(self, d):
        # Here, the question is when the piperidine rings become two separate molecules.
        ring1 = 17 # C17
        ring2 = 21 # N21

        rxn = d['metadata']['mapped_reaction_smiles'].split(">>")
        prod = Chem.MolFromSmiles(rxn[0])
        reacts = [Chem.MolFromSmiles(r) for r in rxn[1].split(".")]

        if (ring1 in  [a.GetAtomMapNum() for a in prod.GetAtoms()]) and (ring2 in  [a.GetAtomMapNum() for a in prod.GetAtoms()]):
            for r in reacts:
                if (ring1 in [a.GetAtomMapNum() for a in r.GetAtoms()]) ^ (ring2 in [a.GetAtomMapNum() for a in r.GetAtoms()]):
                    return True
        return False



class SpecificBondBreak_3(BaseScoring):

    def hit_condition(self, d):
        # Here, the question is when the piperidine rings become two separate molecules.
        ring1 = 24 # N24
        ring2 = 26 # C26

        rxn = d['metadata']['mapped_reaction_smiles'].split(">>")
        prod = Chem.MolFromSmiles(rxn[0])
        reacts = [Chem.MolFromSmiles(r) for r in rxn[1].split(".")]

        if (ring1 in  [a.GetAtomMapNum() for a in prod.GetAtoms()]) and (ring2 in  [a.GetAtomMapNum() for a in prod.GetAtoms()]):
            for r in reacts:
                if (ring1 in [a.GetAtomMapNum() for a in r.GetAtoms()]) ^ (ring2 in [a.GetAtomMapNum() for a in r.GetAtoms()]):
                    return True
        return False


class SpecificBondBreak_4(BaseScoring):

    def hit_condition(self, d):
        # Here, the question is when the piperidine rings become two separate molecules.
        ring1 = 36 # C36
        ring2 = 38 # C38

        rxn = d['metadata']['mapped_reaction_smiles'].split(">>")
        prod = Chem.MolFromSmiles(rxn[0])
        reacts = [Chem.MolFromSmiles(r) for r in rxn[1].split(".")]

        if (ring1 in  [a.GetAtomMapNum() for a in prod.GetAtoms()]) and (ring2 in  [a.GetAtomMapNum() for a in prod.GetAtoms()]):
            for r in reacts:
                if (ring1 in [a.GetAtomMapNum() for a in r.GetAtoms()]) ^ (ring2 in [a.GetAtomMapNum() for a in r.GetAtoms()]):
                    return True
        return False