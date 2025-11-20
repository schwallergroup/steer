"""Generated evaluation code for: Early triazolone ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class TriazoloneRingFormation(BaseScoring):
    """
    Evaluates whether triazolone ring formation occurs early in the synthesis route.
    Checks for the formation of the triazolone heterocycle (c1n[nH]c(=O)[nH]1) and
    scores based on how early this cyclodehydration reaction occurs.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]
        self.direction = config["parameters"]["direction"]
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        
        if self.timing == "early":
            return 1 - x  # Earlier formation is better (lower depth fraction)
        elif self.timing == "late":
            return x  # Later formation is better (higher depth fraction)
        else:
            return 1 if x >= 0 else 0  # Just check if it happens
    
    def hit_condition(self, d):
        """Check if triazolone ring formation occurs in this reaction step."""
        if "mapped_reaction_smiles" not in d.get("metadata", {}):
            return False
            
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
            
            if None in reactants or None in products:
                return False
        except:
            return False
        
        # Check for ring formation: triazolone present in products but not reactants
        if self.direction == "formation":
            reactant_has_ring = any(mol.HasSubstructMatch(self.ring_pattern) for mol in reactants)
            product_has_ring = any(mol.HasSubstructMatch(self.ring_pattern) for mol in products)
            
            return product_has_ring and not reactant_has_ring
        
        # Check for ring breaking: triazolone present in reactants but not products
        elif self.direction == "breaking":
            reactant_has_ring = any(mol.HasSubstructMatch(self.ring_pattern) for mol in reactants)
            product_has_ring = any(mol.HasSubstructMatch(self.ring_pattern) for mol in products)
            
            return reactant_has_ring and not product_has_ring
        
        return False
