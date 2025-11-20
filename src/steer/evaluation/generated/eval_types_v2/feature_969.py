"""Generated evaluation code for: Late stage triazole ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageTriazoleFormation(BaseScoring):
    """
    Evaluates whether triazole ring formation occurs in the late stages of synthesis.
    Detects triazole formation reactions and scores based on how late they occur.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config.get("ring_smarts", "c1nnnc1")
        self.timing = config.get("timing", "late")
        self.direction = config.get("direction", "formation")
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # No triazole formation found
        
        if self.timing == "late":
            return 1 - x  # Higher score for later formation (lower depth fraction)
        elif self.timing == "early":
            return x  # Higher score for earlier formation (higher depth fraction)
        else:
            return 1.0  # Just check presence
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves triazole ring formation.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        rxn_parts = mapped_rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        product_smiles = rxn_parts[0]
        reactant_smiles = rxn_parts[1]
        
        try:
            # Parse molecules
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactant_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Create triazole pattern
            triazole_pattern = Chem.MolFromSmarts(self.ring_smarts)
            if not triazole_pattern:
                return False
            
            if self.direction == "formation":
                # Check if triazole is formed: present in product but not in reactants
                product_has_triazole = product.HasSubstructMatch(triazole_pattern)
                reactants_have_triazole = any(r.HasSubstructMatch(triazole_pattern) for r in reactants)
                
                return product_has_triazole and not reactants_have_triazole
                
            elif self.direction == "breaking":
                # Check if triazole is broken: present in reactants but not in product
                product_has_triazole = product.HasSubstructMatch(triazole_pattern)
                reactants_have_triazole = any(r.HasSubstructMatch(triazole_pattern) for r in reactants)
                
                return not product_has_triazole and reactants_have_triazole
                
        except Exception:
            return False
            
        return False
