"""Generated evaluation code for: Late oxadiazole ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateOxadiazoleFormation(BaseScoring):
    """
    Evaluates whether oxadiazole ring formation occurs late in the synthesis route.
    
    The 1,3,4-oxadiazole ring is formed in the final step via condensation of 
    amidoxime with carboxylic acid. Higher scores are given when the ring formation
    happens closer to the end of the synthesis (lower depth fraction).
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config.get("ring_smarts", "c1nnco1")
        self.timing = config.get("timing", "late")
        self.direction = config.get("direction", "formation")
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        
        if self.timing == "late":
            # Late formation is preferred, so lower depth fraction = higher score
            return (1 - x) * 10
        else:
            # Early formation preference would be x * 10
            return x * 10
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves oxadiazole ring formation.
        For formation, the ring should be present in products but absent in reactants.
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
                
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse products
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            products = [mol for mol in products if mol is not None]
            
            # Parse reactants
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            reactants = [mol for mol in reactants if mol is not None]
            
            if not products or not reactants:
                return False
            
            # Check for ring formation: ring present in products but not in reactants
            if self.direction == "formation":
                has_ring_in_products = any(mol.HasSubstructMatch(self.ring_pattern) for mol in products)
                has_ring_in_reactants = any(mol.HasSubstructMatch(self.ring_pattern) for mol in reactants)
                
                return has_ring_in_products and not has_ring_in_reactants
            
            # For ring breaking (opposite case)
            elif self.direction == "breaking":
                has_ring_in_products = any(mol.HasSubstructMatch(self.ring_pattern) for mol in products)
                has_ring_in_reactants = any(mol.HasSubstructMatch(self.ring_pattern) for mol in reactants)
                
                return not has_ring_in_products and has_ring_in_reactants
                
        except Exception:
            return False
            
        return False
