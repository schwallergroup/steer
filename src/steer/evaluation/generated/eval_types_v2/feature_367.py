"""Generated evaluation code for: Early indole ring formation via Reissert synthesis"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlyIndoleFormation(BaseScoring):
    """
    Evaluates routes for early indole ring formation via Reissert synthesis.
    Checks if an indole ring is formed at or before a specified depth in the synthesis route.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]
        self.formation_depth = config["parameters"]["formation_depth"]
        self.indole_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Indole formation doesn't happen
        
        if self.timing == "early":
            # Reward early formation, penalize late formation
            if x <= self.formation_depth / 10.0:  # Convert depth to fraction
                return 10  # Perfect score for early formation
            else:
                # Linear penalty for late formation
                penalty = (x - self.formation_depth / 10.0) * 10
                return max(0, 10 - penalty)
        else:
            # For other timing preferences, use distance from target
            target_fraction = self.formation_depth / 10.0
            return max(0, 10 - abs(x - target_fraction) * 20)
    
    def hit_condition(self, d):
        """
        Check if indole ring formation occurs in this reaction step.
        """
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            rxn_parts = rxn_smiles.split(">>")
            
            if len(rxn_parts) != 2:
                return False
                
            reactants_smiles = rxn_parts[0]
            products_smiles = rxn_parts[1]
            
            # Parse reactants and products
            reactant_mols = [Chem.MolFromSmiles(smi) for smi in reactants_smiles.split(".")]
            product_mols = [Chem.MolFromSmiles(smi) for smi in products_smiles.split(".")]
            
            # Remove None molecules (parsing failures)
            reactant_mols = [mol for mol in reactant_mols if mol is not None]
            product_mols = [mol for mol in product_mols if mol is not None]
            
            # Count indole rings in reactants and products
            reactant_indole_count = sum(
                len(mol.GetSubstructMatches(self.indole_pattern)) 
                for mol in reactant_mols
            )
            
            product_indole_count = sum(
                len(mol.GetSubstructMatches(self.indole_pattern)) 
                for mol in product_mols
            )
            
            # Indole formation occurs if product has more indole rings than reactants
            return product_indole_count > reactant_indole_count
            
        except (KeyError, AttributeError, ValueError):
            return False
