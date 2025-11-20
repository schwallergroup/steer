"""Generated evaluation code for: Early stage purine ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlyPurineRingFormation(BaseScoring):
    """
    Evaluates whether purine ring formation occurs early in the synthetic route.
    Returns higher scores when purine rings are formed in early stages of synthesis.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]  # "early"
        self.direction = config["parameters"]["direction"]  # "formation"
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        else:
            if self.timing == "early":
                return 1 - x  # Earlier formation gets higher score
            else:
                return x  # Later formation gets higher score
    
    def hit_condition(self, d) -> bool:
        """
        Check if purine ring formation occurs in this reaction step.
        Compares ring count before and after reaction.
        """
        try:
            rxn = d["metadata"]["mapped_reaction_smiles"].split(">>")
            reactants_smiles = rxn[0].split(".")
            products_smiles = rxn[1].split(".")
            
            # Count purine rings in reactants
            reactant_ring_count = 0
            for smi in reactants_smiles:
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    reactant_ring_count += len(mol.GetSubstructMatches(self.ring_pattern))
            
            # Count purine rings in products
            product_ring_count = 0
            for smi in products_smiles:
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    product_ring_count += len(mol.GetSubstructMatches(self.ring_pattern))
            
            # Check for ring formation (increase in ring count)
            if self.direction == "formation":
                return product_ring_count > reactant_ring_count
            else:  # ring breaking
                return reactant_ring_count > product_ring_count
                
        except (KeyError, IndexError, AttributeError):
            return False
