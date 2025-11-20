"""Generated evaluation code for: Late stage epoxide formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageEpoxideFormation(BaseScoring):
    """
    Evaluates whether epoxide ring formation occurs at a late stage in the synthesis route.
    Uses BFS to find the depth at which an epoxide (3-membered ring with oxygen) is formed.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config.get("ring_smarts", "C1CO1")
        self.timing = config.get("timing", "late")
        self.direction = config.get("direction", "formation")
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Epoxide formation doesn't occur
        else:
            # Late stage formation is better (lower depth fraction)
            if self.timing == "late":
                return 1 - x  # Reward later formation
            else:
                return x  # Reward earlier formation
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction involves epoxide formation"""
        metadata = d.get("metadata", {})
        if "mapped_reaction_smiles" not in metadata:
            return False
            
        rxn_smiles = metadata["mapped_reaction_smiles"]
        rxn_parts = rxn_smiles.split(">>")
        
        if len(rxn_parts) != 2:
            return False
            
        reactants_smiles = rxn_parts[0]
        products_smiles = rxn_parts[1]
        
        try:
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            # Filter out None molecules
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            if not reactants or not products:
                return False
            
            # Create epoxide pattern
            epoxide_pattern = Chem.MolFromSmarts(self.ring_smarts)
            if epoxide_pattern is None:
                return False
            
            # Count epoxides in reactants and products
            reactant_epoxides = sum(len(mol.GetSubstructMatches(epoxide_pattern)) 
                                  for mol in reactants)
            product_epoxides = sum(len(mol.GetSubstructMatches(epoxide_pattern)) 
                                 for mol in products)
            
            # Check for epoxide formation (more epoxides in products than reactants)
            if self.direction == "formation":
                return product_epoxides > reactant_epoxides
            else:  # ring breaking
                return reactant_epoxides > product_epoxides
                
        except Exception:
            return False
