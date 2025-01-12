"""Analyze multiple reactions in tree."""

from typing import Tuple

from rdkit import Chem


class MultiRxnCondBase:
    """This class is a bit different because it needs to check multiple reactions."""

    def __call__(self, data):
        """For all routes found (and scored) in the data, find the depth at which the hit condition is met, and plot."""

        score, lengths, lmscore = [], [], []
        for d in data:
            cond, length = self.condition_depth(d["children"][0])
            score.append(10 if cond else 0)
            lengths.append(length)
            lmscore.append(d["lmdata"]["routescore"])
        return score, lmscore  # lengths

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
        prod = Chem.MolFromSmiles(rxn.split(">>")[0])
        reac = Chem.MolFromSmiles(rxn.split(">>")[1])
        return prod.HasSubstructMatch(p) and not reac.HasSubstructMatch(p)

    def get_rxns(self, d):
        def _extract_reactions(d):
            if "metadata" in d:
                yield d["metadata"]["mapped_reaction_smiles"]
                for c in d["children"]:
                    if "children" in c:
                        for r in _extract_reactions(c["children"][0]):
                            yield r

        reactions = list(_extract_reactions(d))
        return reactions


class MultiRxnCond(MultiRxnCondBase):
    def __init__(self, config):
        self.allow_piperidine = config.get("allow_piperidine") or False
        self.allow_oxoisoindolinone = (
            config.get("allow_oxoisoindolinone") or False
        )
        self.allow_piperidine26diox = (
            config.get("allow_piperidine26diox") or False
        )
        self.allow_pyrimidine = config.get("allow_pyrimidine") or False
        self.allow_piridine = config.get("allow_piridine") or False
        self.allow_piperazine = config.get("allow_piperazine") or False

    def condition_depth(self, d) -> Tuple[bool, int]:
        """Extract all the reactions from tree, and find if condition is met."""
        reactions = self.get_rxns(d)

        oxo = any(self.detect_oxoisoindolinone(r) for r in reactions)
        pipe26 = any(self.detect_pipe26diox(r) for r in reactions)
        pip = any(self.detect_piperidine(r) for r in reactions)
        pyrimidine = any(
            self.detect_specific_break(r, "c1cncnc1") for r in reactions
        )
        piridine = any(
            self.detect_specific_break(r, "c1ccncc1") for r in reactions
        )
        piperazine = any(
            self.detect_specific_break(r, "C1CNCCN1") for r in reactions
        )

        condition = (
            oxo == self.allow_oxoisoindolinone
            and pipe26 == self.allow_piperidine26diox
            and pip == self.allow_piperidine
            and pyrimidine == self.allow_pyrimidine
            and piridine == self.allow_piridine
            and piperazine == self.allow_piperazine
        )

        return condition, len(reactions)
