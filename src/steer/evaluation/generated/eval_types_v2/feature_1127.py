"""Generated evaluation code for: Late stage cyclopropane ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageRingFormation(BaseScoring):
    """
    Evaluates routes based on late-stage cyclopropane ring formation.
    Rewards routes where cyclopropane rings are formed closer to the final product.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]  # "late" or "early"
        self.direction = config["parameters"]["direction"]  # "formation" or "breaking"
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        
        if self.timing == "late":
            return 1 - x  # Late-stage formation is better (lower depth fraction)
        else:  # early
            return x  # Early-stage formation is better (higher depth fraction)
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction involves cyclopropane ring formation"""
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
                
            # Count ring matches in reactants and products
            reactant_rings = sum(len(mol.GetSubstructMatches(self.ring_pattern)) 
                               for mol in reactants if mol is not None)
            product_rings = sum(len(mol.GetSubstructMatches(self.ring_pattern)) 
                              for mol in products if mol is not None)
            
            if self.direction == "formation":
                # Ring formation: more rings in products than reactants
                return product_rings > reactant_rings
            else:  # breaking
                # Ring breaking: more rings in reactants than products
                return reactant_rings > product_rings
                
        except Exception:
            return False
