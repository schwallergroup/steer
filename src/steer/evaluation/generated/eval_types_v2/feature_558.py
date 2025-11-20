"""Generated evaluation code for: Early spirocyclopropane ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlySpirocyclopropaneFormation(BaseScoring):
    """
    Checks if spirocyclopropane ring formation occurs early in the synthesis route.
    Returns higher scores when spirocyclopropane formation happens at or before the target step position.
    """
    
    def __init__(self, config: Dict):
        self.target_step = config["parameters"]["step_position"]
        self.timing = config["parameters"]["timing"]
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Spirocyclopropane formation doesn't happen
        
        if self.timing == "early":
            # Early formation is better - lower depth values get higher scores
            if x <= self.target_step / 10.0:  # Convert step to depth fraction
                return 10  # Perfect score for very early formation
            else:
                # Penalize later formation
                return max(0, 10 - (x * 10 - self.target_step))
        else:
            return abs(x * 10 - self.target_step)
    
    def hit_condition(self, d):
        """
        Detects spirocyclopropane ring formation in a reaction.
        Checks if products contain spirocyclopropane that wasn't present in reactants.
        """
        rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
        if not rxn_smiles or ">>" not in rxn_smiles:
            return False
        
        reactants_smiles, products_smiles = rxn_smiles.split(">>")
        
        # Parse reactants and products
        try:
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            
            if None in reactants or None in products:
                return False
        except:
            return False
        
        # Define spirocyclopropane pattern
        # Spirocyclopropane: a cyclopropane ring sharing one carbon with another ring
        spirocyclopropane_pattern = Chem.MolFromSmarts("[C;R2]1[C;R1][C;R1]1")
        
        if spirocyclopropane_pattern is None:
            return False
        
        # Check if spirocyclopropane is formed (present in products but not reactants)
        has_spiro_in_products = any(mol.HasSubstructMatch(spirocyclopropane_pattern) for mol in products)
        has_spiro_in_reactants = any(mol.HasSubstructMatch(spirocyclopropane_pattern) for mol in reactants)
        
        # Return True if spirocyclopropane is formed in this step
        return has_spiro_in_products and not has_spiro_in_reactants
