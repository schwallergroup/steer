"""Generated evaluation code for: Late stage purine ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageRingFormation(BaseScoring):
    """
    Evaluates whether purine ring formation occurs late in the synthesis route.
    Detects formation of purine bicyclic system via ring-forming reactions.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]
        self.direction = config["parameters"]["direction"]
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        
        if self.timing == "late":
            return 1 - x  # Later formation gets higher score
        elif self.timing == "early":
            return x  # Earlier formation gets higher score
        else:
            return 1 if x >= 0 else 0  # Just check if it happens
    
    def hit_condition(self, d):
        """
        Check if this reaction forms the target ring system.
        Returns True if the ring is formed (present in products but not all reactants).
        """
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse products
            products = [Chem.MolFromSmiles(p) for p in products_smiles.split(".")]
            product_has_ring = any(mol and mol.HasSubstructMatch(self.ring_pattern) for mol in products)
            
            # Parse reactants
            reactants = [Chem.MolFromSmiles(r) for r in reactants_smiles.split(".")]
            
            if self.direction == "formation":
                # Ring formation: ring present in products but not in ALL reactants
                if not product_has_ring:
                    return False
                
                # Check if ring is already fully formed in reactants
                reactants_with_ring = [mol for mol in reactants if mol and mol.HasSubstructMatch(self.ring_pattern)]
                
                # Ring formation occurs if ring appears in product but wasn't complete in reactants
                return len(reactants_with_ring) == 0
                
            elif self.direction == "breaking":
                # Ring breaking: ring present in reactants but not products
                reactant_has_ring = any(mol and mol.HasSubstructMatch(self.ring_pattern) for mol in reactants)
                return reactant_has_ring and not product_has_ring
            
            return False
            
        except Exception:
            return False
