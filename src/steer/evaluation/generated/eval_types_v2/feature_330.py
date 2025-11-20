"""Generated evaluation code for: Early stage triazolone ring assembly"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlyTriazoloneAssembly(BaseScoring):
    """
    Evaluates whether triazolone ring formation occurs in early stages of synthesis.
    
    Detects formation of triazolone rings ([#7]1[#7][#6][#7][#6]1=O) and scores
    based on how early in the route this ring assembly occurs.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.triazolone_pattern = Chem.MolFromSmarts(self.ring_smarts)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        else:
            return x  # Earlier formation (smaller x) gives better score
            
    def hit_condition(self, d) -> bool:
        """Check if triazolone ring is formed in this reaction step."""
        rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
        rxn_parts = rxn_smiles.split(">>")
        
        if len(rxn_parts) != 2:
            return False
            
        reactants_smiles = rxn_parts[0]
        products_smiles = rxn_parts[1]
        
        # Parse reactants and products
        try:
            reactants = [Chem.MolFromSmiles(smi) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi) for smi in products_smiles.split(".")]
            
            if not all(reactants) or not all(products):
                return False
                
        except:
            return False
            
        # Check if triazolone ring is absent in reactants but present in products
        reactant_has_triazolone = any(
            mol.HasSubstructMatch(self.triazolone_pattern) for mol in reactants
        )
        
        product_has_triazolone = any(
            mol.HasSubstructMatch(self.triazolone_pattern) for mol in products  
        )
        
        # Ring formation occurs if absent in reactants but present in products
        return not reactant_has_triazolone and product_has_triazolone
