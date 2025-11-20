"""Generated evaluation code for: Late imidazole ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ImidazoleRingFormation(BaseScoring):
    """
    Evaluates whether imidazole ring formation occurs late in the synthesis route.
    Detects when an imidazole ring (c1cncn1) is formed and rewards later formation.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]  # "c1cncn1"
        self.timing = config["parameters"]["timing"]  # "late" 
        self.direction = config["parameters"]["direction"]  # "formation"
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        else:
            if self.timing == "late":
                return 1 - x  # Later formation gets higher score
            else:  # early timing
                return x  # Earlier formation gets higher score
    
    def hit_condition(self, d):
        """Check if imidazole ring formation occurs in this reaction step"""
        if "mapped_reaction_smiles" not in d.get("metadata", {}):
            return False
            
        rxn = d["metadata"]["mapped_reaction_smiles"].split(">>")
        if len(rxn) != 2:
            return False
            
        prod = Chem.MolFromSmiles(rxn[0])
        reactants = [Chem.MolFromSmiles(r) for r in rxn[1].split(".") if r]
        
        if not prod or not all(reactants):
            return False
        
        # Check if product has imidazole ring
        prod_has_ring = prod.HasSubstructMatch(self.ring_pattern)
        
        if self.direction == "formation":
            # Ring formation: product has ring but no reactant has complete ring
            if not prod_has_ring:
                return False
            
            # Check if any reactant already has the complete imidazole ring
            for reactant in reactants:
                if reactant.HasSubstructMatch(self.ring_pattern):
                    return False  # Ring already existed, not formation
            
            return True  # Product has ring, no reactant had it - formation occurred
            
        elif self.direction == "breaking":
            # Ring breaking: reactant has ring but product doesn't
            reactant_has_ring = any(r.HasSubstructMatch(self.ring_pattern) for r in reactants)
            return reactant_has_ring and not prod_has_ring
        
        return False
