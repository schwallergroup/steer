"""Generated evaluation code for: Late 1,3,4-oxadiazole ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateOxadiazoleFormation(BaseScoring):
    """
    Evaluates whether 1,3,4-oxadiazole ring formation occurs late in the synthesis route.
    
    This scoring function identifies when a 1,3,4-oxadiazole ring (c1nnoc1) is formed
    and rewards late-stage formation, which is typical for cyclodehydration reactions
    of hydrazides with triethyl orthoformate.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config.get("ring_smarts", "c1nnoc1")
        self.timing = config.get("timing", "late")
        self.direction = config.get("direction", "formation")
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
        
    def route_scoring(self, x) -> float:
        """
        Convert depth fraction to score (0-10 scale).
        For late timing: higher depth fraction = better score
        """
        if x < 0:
            return 0  # Ring formation doesn't occur
        
        if self.timing == "late":
            # Late formation is better - reward higher depth fractions
            return 10 * x
        else:
            # Early formation is better - reward lower depth fractions  
            return 10 * (1 - x)
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction node involves 1,3,4-oxadiazole ring formation.
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles:
                return False
                
            parts = rxn_smiles.split(">>")
            if len(parts) != 2:
                return False
                
            product_smiles = parts[0]
            reactant_smiles = parts[1]
            
            product = Chem.MolFromSmiles(product_smiles)
            if not product:
                return False
                
            # Check if product contains the oxadiazole ring
            if not product.HasSubstructMatch(self.ring_pattern):
                return False
            
            # Check if ring is being formed (not present in reactants)
            reactant_mols = []
            for r_smiles in reactant_smiles.split("."):
                r_mol = Chem.MolFromSmiles(r_smiles)
                if r_mol:
                    reactant_mols.append(r_mol)
            
            if self.direction == "formation":
                # Ring formation: present in product but not in any reactant
                for reactant in reactant_mols:
                    if reactant.HasSubstructMatch(self.ring_pattern):
                        return False  # Ring already exists in reactants
                return True  # Ring formed in this step
            
            elif self.direction == "breaking":
                # Ring breaking: present in reactants but not in product
                reactant_has_ring = any(r.HasSubstructMatch(self.ring_pattern) for r in reactant_mols)
                return reactant_has_ring and not product.HasSubstructMatch(self.ring_pattern)
                
        except Exception:
            return False
            
        return False
