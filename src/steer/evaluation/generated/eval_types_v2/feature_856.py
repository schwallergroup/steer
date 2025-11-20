"""Generated evaluation code for: Late stage piperidine cyclization"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageRingFormation(BaseScoring):
    """
    Evaluates whether a specific ring structure is formed late in the synthesis route.
    Higher scores are given when the target ring is formed closer to the final product.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]  # "late" or "early"
        self.direction = config["parameters"]["direction"]  # "formation" or "break"
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        
        if self.timing == "late":
            return 1 - x  # Later formation gets higher score
        else:  # early
            return x  # Earlier formation gets higher score
    
    def hit_condition(self, d):
        """Check if this reaction involves formation/breaking of the target ring"""
        rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
        rxn_parts = rxn_smiles.split(">>")
        
        if len(rxn_parts) != 2:
            return False
            
        reactants_smiles = rxn_parts[0]
        products_smiles = rxn_parts[1]
        
        # Parse reactants and products
        reactants = []
        for r_smiles in reactants_smiles.split("."):
            mol = Chem.MolFromSmiles(r_smiles)
            if mol:
                reactants.append(mol)
        
        products = []
        for p_smiles in products_smiles.split("."):
            mol = Chem.MolFromSmiles(p_smiles)
            if mol:
                products.append(mol)
        
        # Count ring matches in reactants and products
        reactant_matches = sum(1 for mol in reactants if mol.HasSubstructMatch(self.ring_pattern))
        product_matches = sum(1 for mol in products if mol.HasSubstructMatch(self.ring_pattern))
        
        if self.direction == "formation":
            # Ring formation: fewer rings in reactants than products
            return product_matches > reactant_matches
        else:  # "break"
            # Ring breaking: more rings in reactants than products
            return reactant_matches > product_matches
