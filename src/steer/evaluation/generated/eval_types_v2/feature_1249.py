"""Generated evaluation code for: Late triazole ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateTriazoleFormation(BaseScoring):
    """
    Evaluates synthesis routes for late-stage triazole ring formation.
    
    Checks if a 1,2,4-triazole ring (c1nnnc1) is formed at or after a specified
    depth threshold, rewarding late-stage heterocycle installation strategies.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.stage = config["parameters"]["stage"] 
        self.depth_threshold = config["parameters"]["depth_threshold"]
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        
        if self.stage == "late":
            # For late-stage, reward formations at or after depth threshold
            if x >= self.depth_threshold / 10.0:  # x is depth fraction
                return 1.0  # Perfect score for late formation
            else:
                return x * 10 / self.depth_threshold  # Partial score for earlier formation
        else:
            # For early-stage, reward formations before depth threshold  
            if x <= self.depth_threshold / 10.0:
                return 1.0
            else:
                return max(0, 1.0 - (x - self.depth_threshold / 10.0) * 2)
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction forms a triazole ring."""
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            rxn_parts = rxn_smiles.split(">>")
            
            if len(rxn_parts) != 2:
                return False
                
            reactants_smiles = rxn_parts[0]
            products_smiles = rxn_parts[1]
            
            # Parse reactants and products
            reactants = []
            for smi in reactants_smiles.split("."):
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    reactants.append(mol)
            
            products = []
            for smi in products_smiles.split("."):
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    products.append(mol)
            
            # Count triazole rings before and after reaction
            reactant_triazole_count = sum(
                len(mol.GetSubstructMatches(self.ring_pattern)) 
                for mol in reactants
            )
            
            product_triazole_count = sum(
                len(mol.GetSubstructMatches(self.ring_pattern))
                for mol in products  
            )
            
            # Ring formation occurs if products have more triazole rings than reactants
            return product_triazole_count > reactant_triazole_count
            
        except (KeyError, AttributeError, ValueError):
            return False
