"""Generated evaluation code for: Early quinolinone core assembly"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlyQuinolinoneAssembly(BaseScoring):
    """
    Evaluates whether quinolinone core assembly occurs early in the synthesis route.
    Checks if the quinolinone bicyclic core is formed at the specified early step position.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]
        self.step_position = config["parameters"]["step_position"]
        self.quinolinone_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Quinolinone core not formed
        
        if self.timing == "early":
            # Reward formation at or before target step position
            if x <= self.step_position / 10.0:  # Convert step to depth fraction
                return 10  # Maximum score for early formation
            else:
                # Penalize later formation, score decreases with depth
                return max(0, 10 - (x * 10 - self.step_position) * 2)
        
        return 0
    
    def hit_condition(self, d):
        """
        Check if quinolinone core is formed in this reaction step.
        """
        rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
        if not rxn_smiles:
            return False
            
        parts = rxn_smiles.split(">>")
        if len(parts) != 2:
            return False
            
        product_smiles = parts[0]
        reactant_smiles = parts[1]
        
        try:
            product_mol = Chem.MolFromSmiles(product_smiles)
            if not product_mol:
                return False
                
            # Check if product contains quinolinone core
            product_has_quinolinone = product_mol.HasSubstructMatch(self.quinolinone_pattern)
            
            if not product_has_quinolinone:
                return False
                
            # Check if any reactant already has the quinolinone core
            reactants = reactant_smiles.split(".")
            for reactant_smile in reactants:
                reactant_mol = Chem.MolFromSmiles(reactant_smile.strip())
                if reactant_mol and reactant_mol.HasSubstructMatch(self.quinolinone_pattern):
                    return False  # Core already existed, not a formation step
                    
            return True  # Quinolinone core formed in this step
            
        except Exception:
            return False
