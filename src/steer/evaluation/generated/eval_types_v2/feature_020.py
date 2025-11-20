"""Generated evaluation code for: Late stage cyclopropane ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageRingFormation(BaseScoring):
    """
    Evaluates synthesis routes for late-stage ring formation reactions.
    Specifically checks for cyclopropane ring formation timing in the synthetic route.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["ring_smarts"]  # "C1CC1" for cyclopropane
        self.timing = config["timing"]  # "late"
        self.direction = config["direction"]  # "formation"
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        else:
            # For late-stage preference, higher depth fraction is better
            if self.timing == "late":
                return x * 10  # Convert to 0-10 scale, favoring late stage
            elif self.timing == "early":
                return (1 - x) * 10  # Favor early stage
            else:
                return 5  # Neutral if no timing preference
    
    def hit_condition(self, d):
        """Check if this reaction involves the specified ring formation/breaking."""
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            # Remove None values from failed parsing
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            if not reactants or not products:
                return False
            
            # Count ring instances in reactants and products
            reactant_ring_count = sum(len(mol.GetSubstructMatches(self.ring_pattern)) 
                                    for mol in reactants)
            product_ring_count = sum(len(mol.GetSubstructMatches(self.ring_pattern)) 
                                   for mol in products)
            
            # Check for ring formation or breaking based on direction
            if self.direction == "formation":
                return product_ring_count > reactant_ring_count
            elif self.direction == "breaking":
                return reactant_ring_count > product_ring_count
            else:
                # Any change in ring count
                return reactant_ring_count != product_ring_count
                
        except (KeyError, ValueError, AttributeError):
            return False
