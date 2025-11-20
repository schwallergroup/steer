"""Generated evaluation code for: Early stage cyclopropane formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlyCyclopropaneFormation(BaseScoring):
    """
    Evaluates synthesis routes for early stage cyclopropane formation.
    Checks if a cyclopropane ring is formed within the first few steps of synthesis.
    """
    
    def __init__(self, config: Dict):
        self.step_position = config.get("step_position", 10)
        self.timing = config.get("timing", "early")
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Cyclopropane formation doesn't happen
        
        if self.timing == "early":
            # Early stage formation is preferred - lower depth is better
            if x <= 0.1:  # Very early (first 10% of steps)
                return 10
            elif x <= 0.3:  # Reasonably early
                return 8
            elif x <= 0.5:  # Mid-stage
                return 5
            else:  # Late stage
                return 2
        else:
            # Generic scoring - penalize deviation from target position
            target_fraction = self.step_position / 100.0
            return max(0, 10 - 10 * abs(x - target_fraction))
    
    def hit_condition(self, d) -> bool:
        """
        Check if the reaction forms a cyclopropane ring.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
        
        try:
            rxn_parts = mapped_rxn.split(">>")
            reactants_smiles = rxn_parts[0]
            products_smiles = rxn_parts[1]
            
            # Parse reactants and products
            reactants = []
            for r_smi in reactants_smiles.split("."):
                mol = Chem.MolFromSmiles(r_smi.strip())
                if mol:
                    reactants.append(mol)
            
            products = []
            for p_smi in products_smiles.split("."):
                mol = Chem.MolFromSmiles(p_smi.strip())
                if mol:
                    products.append(mol)
            
            if not reactants or not products:
                return False
            
            # Count cyclopropane rings in reactants and products
            cyclopropane_pattern = Chem.MolFromSmarts("[#6]1[#6][#6]1")  # 3-membered carbon ring
            
            reactant_cyclopropanes = sum(
                len(mol.GetSubstructMatches(cyclopropane_pattern)) 
                for mol in reactants
            )
            
            product_cyclopropanes = sum(
                len(mol.GetSubstructMatches(cyclopropane_pattern)) 
                for mol in products
            )
            
            # Cyclopropane formation occurs if products have more cyclopropanes than reactants
            return product_cyclopropanes > reactant_cyclopropanes
            
        except Exception:
            return False
