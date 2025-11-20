"""Generated evaluation code for: Late cyclopropane ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class CyclopropaneFormationTiming(BaseScoring):
    """
    Evaluates the timing of cyclopropane ring formation in synthesis routes.
    Rewards late-stage cyclopropane formation, typically via reactions like
    Corey-Chaykovsky cyclopropanation.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config.get("ring_smarts", "C1CC1")
        self.timing = config.get("timing", "late")  # "early" or "late"
        self.direction = config.get("direction", "formation")  # "formation" or "break"
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't occur
        
        if self.timing == "late":
            # Late formation preferred - higher score for later depth
            return 1 - x
        else:
            # Early formation preferred - higher score for earlier depth
            return x
    
    def hit_condition(self, d) -> bool:
        """
        Check if cyclopropane ring formation/breaking occurs in this reaction step.
        """
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            # Remove None molecules from parsing failures
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            # Create cyclopropane pattern
            cyclopropane_pattern = Chem.MolFromSmarts(self.ring_smarts)
            if cyclopropane_pattern is None:
                return False
            
            # Count cyclopropane rings in reactants and products
            reactant_rings = sum(len(mol.GetSubstructMatches(cyclopropane_pattern)) 
                               for mol in reactants)
            product_rings = sum(len(mol.GetSubstructMatches(cyclopropane_pattern)) 
                              for mol in products)
            
            if self.direction == "formation":
                # Ring formation: more rings in products than reactants
                return product_rings > reactant_rings
            else:
                # Ring breaking: more rings in reactants than products
                return reactant_rings > product_rings
                
        except (KeyError, ValueError, AttributeError):
            return False
