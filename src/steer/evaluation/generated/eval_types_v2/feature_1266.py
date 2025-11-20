"""Generated evaluation code for: Late pyrrolidine ring formation via intramolecular cyclization"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LatePyrrolidineFormation(BaseScoring):
    """
    Evaluates synthesis routes based on late-stage pyrrolidine ring formation 
    via intramolecular cyclization. Rewards routes where pyrrolidine rings 
    are formed in the final stages through intramolecular cyclization.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config.get("ring_smarts", "C1CCNC1")
        self.timing = config.get("timing", "late")
        self.formation_method = config.get("formation_method", "intramolecular")
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)

    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        else:
            # Late-stage formation is better (higher depth fraction = later)
            return x * 10

    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves pyrrolidine ring formation via 
        intramolecular cyclization.
        """
        try:
            rxn = d["metadata"]["mapped_reaction_smiles"].split(">>")
            prod = Chem.MolFromSmiles(rxn[0])
            reactants = [Chem.MolFromSmiles(r) for r in rxn[1].split(".")]
            
            if not prod or not all(reactants):
                return False
            
            # Check if pyrrolidine ring is formed (present in product but not reactants)
            prod_has_pyrrolidine = prod.HasSubstructMatch(self.ring_pattern)
            reactants_have_pyrrolidine = any(r.HasSubstructMatch(self.ring_pattern) for r in reactants)
            
            if not (prod_has_pyrrolidine and not reactants_have_pyrrolidine):
                return False
            
            # Check for intramolecular cyclization (single reactant forms the ring)
            if self.formation_method == "intramolecular":
                return len(reactants) == 1 and self._is_cyclization_precursor(reactants[0], prod)
            
            return True
            
        except Exception:
            return False

    def _is_cyclization_precursor(self, reactant, product) -> bool:
        """
        Check if the reactant is a suitable precursor for intramolecular 
        pyrrolidine formation by comparing atom counts and connectivity.
        """
        try:
            # Basic check: reactant should have similar heavy atom count
            reactant_atoms = reactant.GetNumHeavyAtoms()
            product_atoms = product.GetNumHeavyAtoms()
            
            # Allow for minor differences due to leaving groups/protecting groups
            if abs(reactant_atoms - product_atoms) > 3:
                return False
            
            # Check if reactant contains the open-chain precursor pattern
            # Common patterns for pyrrolidine formation
            precursor_patterns = [
                "NCCCC",  # Amino-butyl chain
                "N(C)CCCC",  # N-methyl amino-butyl
                "NCCC(C)C",  # Branched amino-butyl
                "NCCCC(=O)",  # Amino-butyl with carbonyl
            ]
            
            for pattern_smarts in precursor_patterns:
                pattern = Chem.MolFromSmarts(pattern_smarts)
                if pattern and reactant.HasSubstructMatch(pattern):
                    return True
            
            return False
            
        except Exception:
            return False
