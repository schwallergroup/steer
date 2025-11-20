"""Generated evaluation code for: Early quinoline ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class QuinolineRingFormation(BaseScoring):
    """Evaluates early quinoline ring formation in synthesis routes.
    
    Checks if quinoline ring formation occurs early in the synthesis,
    rewarding routes where the quinoline core is constructed before
    sidechain elaboration.
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
            return 1 - x  # Earlier formation gets higher score
        elif self.timing == "late":
            return x  # Later formation gets higher score
        else:
            return 0.5  # Neutral if timing not specified
    
    def hit_condition(self, d) -> bool:
        """Check if quinoline ring formation occurs in this reaction step."""
        try:
            rxn = d["metadata"]["mapped_reaction_smiles"].split(">>")
            reactants_smiles = rxn[0].split(".")
            product_smiles = rxn[1]
            
            # Parse molecules
            reactants = [Chem.MolFromSmiles(r) for r in reactants_smiles if r]
            product = Chem.MolFromSmiles(product_smiles)
            
            if not product or not all(reactants):
                return False
            
            # Check if product has quinoline ring
            product_has_quinoline = product.HasSubstructMatch(self.ring_pattern)
            
            if not product_has_quinoline:
                return False
            
            # Check if any reactant already has complete quinoline ring
            reactants_have_quinoline = any(
                r.HasSubstructMatch(self.ring_pattern) for r in reactants
            )
            
            if self.direction == "formation":
                # Ring formation: product has quinoline but reactants don't
                return product_has_quinoline and not reactants_have_quinoline
            elif self.direction == "breaking":
                # Ring breaking: reactants have quinoline but product doesn't
                return reactants_have_quinoline and not product_has_quinoline
            else:
                # Just check presence in product
                return product_has_quinoline
                
        except Exception:
            return False
