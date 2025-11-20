"""Generated evaluation code for: Late piperidine ring formation via intramolecular cyclization"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateRingFormationTiming(BaseScoring):
    """
    Evaluates whether a specific ring is formed late in the synthesis route.
    Checks for intramolecular cyclization reactions that form the target ring structure.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]  # "late" or "early"
        self.direction = config["parameters"]["direction"]  # "formation" or "break"
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        
        if self.timing == "late":
            return 1 - x  # Later formation gets higher score
        else:  # early
            return x  # Earlier formation gets higher score
    
    def hit_condition(self, d):
        """Check if this reaction involves formation of the target ring"""
        rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
        if not rxn_smiles or ">>" not in rxn_smiles:
            return False
            
        try:
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            
            # Remove None molecules (parsing failures)
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            if not reactants or not products:
                return False
                
            # Create SMARTS pattern for ring detection
            ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
            if ring_pattern is None:
                return False
            
            # Check for ring formation: ring present in products but not in reactants
            if self.direction == "formation":
                # Count rings in products
                rings_in_products = sum(1 for prod in products if prod.HasSubstructMatch(ring_pattern))
                
                # Count rings in reactants
                rings_in_reactants = sum(1 for react in reactants if react.HasSubstructMatch(ring_pattern))
                
                # Ring formation occurs if more rings in products than reactants
                return rings_in_products > rings_in_reactants
                
            else:  # direction == "break"
                # Count rings in reactants
                rings_in_reactants = sum(1 for react in reactants if react.HasSubstructMatch(ring_pattern))
                
                # Count rings in products  
                rings_in_products = sum(1 for prod in products if prod.HasSubstructMatch(ring_pattern))
                
                # Ring breaking occurs if more rings in reactants than products
                return rings_in_reactants > rings_in_products
                
        except Exception:
            return False
