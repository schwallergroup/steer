"""Analyze multiple reactions in tree."""

from rdkit import Chem
from typing import Tuple

class MultiRxnCondBase:
    """This class is a bit different because it needs to check multiple reactions."""
    def __call__(self, data):
        """For all routes found (and scored) in the data, find the depth at which the hit condition is met, and plot."""
        
        conds, lengths, score = [], [], []
        for d in data:
            cond, length = self.condition_depth(d['children'][0])
            conds.append(cond)
            lengths.append(length)
            score.append(d['lmdata']['routescore'])
        return conds, score # lengths

    def detect_piperidine(self, rxn):
        oxoisoindolinone = "C1CN[CH2]CC1"
        return self.detect_specific_break(rxn, oxoisoindolinone)

    def detect_oxoisoindolinone(self, rxn):
        oxoisoindolinone = "c1cC(=O)NC1"
        return self.detect_specific_break(rxn, oxoisoindolinone)

    def detect_pipe26diox(self, rxn):
        pipe26diox = "NC1CCC(=O)NC1=O"
        return self.detect_specific_break(rxn, pipe26diox)

    def detect_specific_break(self, rxn, pattern):
        p = Chem.MolFromSmarts(pattern)
        prod = Chem.MolFromSmiles(rxn.split('>>')[0])
        reac = Chem.MolFromSmiles(rxn.split('>>')[1])
        return prod.HasSubstructMatch(p) and not reac.HasSubstructMatch(p)

    def get_rxns(self, d):
        def _extract_reactions(d):
            if 'metadata' in d:
                yield d['metadata']['mapped_reaction_smiles']
                for c in d['children']:
                    if 'children' in c:
                        for r in _extract_reactions(c['children'][0]):
                            yield r
        reactions = list(_extract_reactions(d))
        return reactions


class MultiRxnCond(MultiRxnCondBase):
    def __init__(self, config):
        self.allow_piperidine = config["allow_piperidine"]
        self.allow_oxoisoindolinone = config["allow_oxoisoindolinone"]
        self.allow_piperidine26diox = config["allow_piperidine26diox"]

    def condition_depth(self, d) -> Tuple[bool, int]:
        """Extract all the reactions from tree, and find if condition is met."""
        reactions = self.get_rxns(d)

        oxo = any(self.detect_oxoisoindolinone(r) for r in reactions)
        pipe26 = any(self.detect_pipe26diox(r) for r in reactions)
        pip = any(self.detect_piperidine(r) for r in reactions)

        condition = (
            oxo == self.allow_oxoisoindolinone and 
            pipe26 == self.allow_piperidine26diox and 
            pip == self.allow_piperidine
        )

        return condition, len(reactions)
