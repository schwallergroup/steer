"""Generated evaluation code for: Late stage lactam ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageRingFormation(BaseScoring):
    """
    Evaluates synthesis routes for late-stage ring formation reactions.
    Checks if a specific ring pattern is formed at a particular timing in the route.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]  # "late", "early", "any"
        self.direction = config["parameters"]["direction"]  # "formation", "breaking"
        
        # Compile the SMARTS pattern
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
        if self.ring_pattern is None:
            raise ValueError(f"Invalid SMARTS pattern: {self.ring_smarts}")
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        
        if self.timing == "late":
            return 1 - x  # Later is better, score decreases with earlier timing
        elif self.timing == "early":
            return x  # Earlier is better, score increases with later timing
        else:  # "any"
            return 1  # Just needs to happen somewhere
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction node involves the specified ring formation/breaking.
        """
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse molecules
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            
            # Filter out None molecules
            products = [mol for mol in products if mol is not None]
            reactants = [mol for mol in reactants if mol is not None]
            
            # Count ring pattern occurrences in products and reactants
            product_matches = sum(len(mol.GetSubstructMatches(self.ring_pattern)) for mol in products)
            reactant_matches = sum(len(mol.GetSubstructMatches(self.ring_pattern)) for mol in reactants)
            
            if self.direction == "formation":
                # Ring formation: more rings in products than reactants
                return product_matches > reactant_matches
            else:  # "breaking"
                # Ring breaking: fewer rings in products than reactants
                return reactant_matches > product_matches
                
        except (KeyError, ValueError, AttributeError):
            return False
