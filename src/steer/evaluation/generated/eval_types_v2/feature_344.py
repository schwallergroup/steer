"""Generated evaluation code for: Late triazole ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateTriazoleFormation(BaseScoring):
    """
    Evaluates whether triazole ring formation occurs late in the synthesis route.
    Detects formation of triazole rings (c1nn[nH]c1 pattern) and scores based on timing.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config.get("ring_smarts", "c1nn[nH]c1")
        self.timing = config.get("timing", "late")
        self.direction = config.get("direction", "formation")
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        
        if self.timing == "late":
            # Later formation (higher depth fraction) gets better score
            return 10 * (1 - x)
        elif self.timing == "early":
            # Earlier formation (lower depth fraction) gets better score  
            return 10 * x
        else:
            # For specific timing, penalize deviation
            target_depth = float(self.timing)
            return max(0, 10 - 10 * abs(x - target_depth))
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction forms a triazole ring"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles")
        
        if not mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            product_smiles = rxn_parts[0]
            reactant_smiles = rxn_parts[1]
            
            product_mol = Chem.MolFromSmiles(product_smiles)
            if not product_mol:
                return False
                
            # Check if product contains triazole pattern
            triazole_pattern = Chem.MolFromSmarts(self.ring_smarts)
            if not triazole_pattern:
                return False
                
            product_has_triazole = product_mol.HasSubstructMatch(triazole_pattern)
            
            if self.direction == "formation":
                if not product_has_triazole:
                    return False
                    
                # Check that reactants don't have the triazole ring
                reactants = reactant_smiles.split(".")
                for reactant_smiles in reactants:
                    reactant_mol = Chem.MolFromSmiles(reactant_smiles)
                    if reactant_mol and reactant_mol.HasSubstructMatch(triazole_pattern):
                        return False  # Ring already present in reactant
                        
                return True  # Ring formed in this step
                
            elif self.direction == "breaking":
                # Check if reactants have triazole but product doesn't
                if product_has_triazole:
                    return False
                    
                reactants = reactant_smiles.split(".")
                for reactant_smiles in reactants:
                    reactant_mol = Chem.MolFromSmiles(reactant_smiles)
                    if reactant_mol and reactant_mol.HasSubstructMatch(triazole_pattern):
                        return True  # Ring was broken in this step
                        
                return False
                
        except Exception:
            return False
            
        return False
