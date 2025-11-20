"""Generated evaluation code for: Late pyridine ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LatePyridineRingFormation(BaseScoring):
    """
    Checks if pyridine ring formation occurs late in the synthesis route.
    Returns higher scores for pyridine ring formation that happens closer to the final step.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]  # "c1ccncc1" for pyridine
        self.timing = config["parameters"]["timing"]  # "late"
        self.direction = config["parameters"]["direction"]  # "formation"
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)

    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        else:
            if self.timing == "late":
                return 1 - x  # Later formation gets higher score (closer to 1.0)
            else:
                return x  # Earlier formation gets higher score

    def hit_condition(self, d) -> bool:
        """Check if this reaction involves pyridine ring formation"""
        rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
        if not rxn_smiles or ">>" not in rxn_smiles:
            return False
            
        reactants_smiles, products_smiles = rxn_smiles.split(">>")
        
        # Parse reactants and products
        reactants = []
        for r_smi in reactants_smiles.split("."):
            mol = Chem.MolFromSmiles(r_smi)
            if mol:
                reactants.append(mol)
        
        products = []
        for p_smi in products_smiles.split("."):
            mol = Chem.MolFromSmiles(p_smi)
            if mol:
                products.append(mol)
        
        if not reactants or not products:
            return False
        
        # Count pyridine rings in reactants and products
        reactant_rings = sum(len(mol.GetSubstructMatches(self.ring_pattern)) for mol in reactants)
        product_rings = sum(len(mol.GetSubstructMatches(self.ring_pattern)) for mol in products)
        
        if self.direction == "formation":
            # Ring formation: more rings in products than reactants
            return product_rings > reactant_rings
        elif self.direction == "breaking":
            # Ring breaking: more rings in reactants than products
            return reactant_rings > product_rings
        
        return False
