"""Generated evaluation code for: Late stage pyrazole ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageRingFormation(BaseScoring):
    """
    Evaluates late-stage ring formation for a specific ring pattern.
    Rewards routes where the target ring is formed later in the synthesis sequence.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]  # "late" or "early"
        self.direction = config["parameters"]["direction"]  # "formation" or "breaking"
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
        
    def route_scoring(self, x) -> float:
        """Convert depth fraction to score (0-10 scale)"""
        if x < 0:
            return 0  # Ring formation doesn't happen
        
        if self.timing == "late":
            # Later formation is better - higher depth fraction gives higher score
            return x * 10
        else:  # early
            # Earlier formation is better - lower depth fraction gives higher score
            return (1 - x) * 10
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction involves the target ring formation/breaking"""
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
                
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse molecules
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".") if p.strip()]
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".") if r.strip()]
            
            # Filter out None molecules
            products = [mol for mol in products if mol is not None]
            reactants = [mol for mol in reactants if mol is not None]
            
            if not products or not reactants:
                return False
            
            # Count ring pattern matches in products and reactants
            product_matches = sum(len(mol.GetSubstructMatches(self.ring_pattern)) for mol in products)
            reactant_matches = sum(len(mol.GetSubstructMatches(self.ring_pattern)) for mol in reactants)
            
            if self.direction == "formation":
                # Ring formation: more rings in products than reactants
                return product_matches > reactant_matches
            else:  # breaking
                # Ring breaking: fewer rings in products than reactants
                return product_matches < reactant_matches
                
        except Exception:
            return False
