"""Generated evaluation code for: Late stage spiro-cyclopropane ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageCyclopropaneFormation(BaseScoring):
    """
    Evaluates whether cyclopropane ring formation occurs at a late stage in the synthesis.
    Returns higher scores when cyclopropane formation happens closer to the final product.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.stage_threshold = config["parameters"]["stage_threshold"]
        self.cyclopropane_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Cyclopropane formation doesn't happen
        
        # For late-stage preference, score higher when x approaches 1
        if x >= self.stage_threshold:
            return 10  # Perfect late-stage formation
        else:
            # Linear scaling: earlier formation gets lower scores
            return x * 10
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves cyclopropane ring formation.
        """
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(smi) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi) for smi in products_smiles.split(".")]
            
            if not all(reactants) or not all(products):
                return False
            
            # Count cyclopropane rings in reactants and products
            reactant_cyclopropanes = sum(
                len(mol.GetSubstructMatches(self.cyclopropane_pattern)) 
                for mol in reactants
            )
            
            product_cyclopropanes = sum(
                len(mol.GetSubstructMatches(self.cyclopropane_pattern)) 
                for mol in products
            )
            
            # Ring formation: more cyclopropane rings in products than reactants
            return product_cyclopropanes > reactant_cyclopropanes
            
        except Exception:
            return False
