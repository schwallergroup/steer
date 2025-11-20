"""Generated evaluation code for: Late oxadiazolone ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateOxadiazoloneFormation(BaseScoring):
    """
    Evaluates whether oxadiazolone ring formation occurs late in the synthesis route.
    Checks for the formation of the 5-membered oxadiazolone ring structure and 
    rewards routes where this cyclization happens in the final steps.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]
        self.direction = config["parameters"]["direction"]
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        else:
            # For late timing, reward formations that happen closer to the end
            # x is depth fraction (0 = root, 1 = leaves)
            if self.timing == "late":
                return x * 10  # Higher score for higher depth fraction
            else:
                return (1 - x) * 10  # Higher score for lower depth fraction
    
    def hit_condition(self, d):
        """Check if this reaction involves oxadiazolone ring formation"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        rxn_parts = mapped_rxn.split(">>")
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        # Parse reactants and products
        reactant_mols = []
        for smi in reactants.split("."):
            mol = Chem.MolFromSmiles(smi)
            if mol:
                reactant_mols.append(mol)
        
        product_mols = []
        for smi in products.split("."):
            mol = Chem.MolFromSmiles(smi)
            if mol:
                product_mols.append(mol)
        
        if not reactant_mols or not product_mols:
            return False
        
        if self.direction == "formation":
            # Check if oxadiazolone ring is formed (absent in reactants, present in products)
            reactants_have_ring = any(mol.HasSubstructMatch(self.ring_pattern) for mol in reactant_mols)
            products_have_ring = any(mol.HasSubstructMatch(self.ring_pattern) for mol in product_mols)
            
            return not reactants_have_ring and products_have_ring
        
        elif self.direction == "breaking":
            # Check if oxadiazolone ring is broken (present in reactants, absent in products)
            reactants_have_ring = any(mol.HasSubstructMatch(self.ring_pattern) for mol in reactant_mols)
            products_have_ring = any(mol.HasSubstructMatch(self.ring_pattern) for mol in product_mols)
            
            return reactants_have_ring and not products_have_ring
        
        return False
