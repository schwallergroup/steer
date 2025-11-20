"""Generated evaluation code for: Late stage cyclopropanation reaction"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageRingFormation(BaseScoring):
    """
    Evaluates whether a specific ring formation occurs at a late stage in the synthesis.
    Checks for the formation of a ring pattern (e.g., cyclopropane) within a specified depth threshold.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]  # "late" or "early"
        self.depth_threshold = config["parameters"]["depth_threshold"]
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        
        if self.timing == "late":
            # For late-stage: reward formation at shallow depths (closer to target)
            if x <= self.depth_threshold / 10.0:  # Convert to depth fraction
                return 10 * (1 - x)  # Higher score for earlier occurrence
            else:
                return 0  # Penalize if too deep
        else:  # "early"
            # For early-stage: reward formation at deeper depths
            return 10 * x
    
    def hit_condition(self, d):
        """
        Check if the specified ring is formed in this reaction step.
        Ring formation is detected by checking if the ring pattern exists in 
        products but not in all reactants.
        """
        rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
        if not rxn_smiles or ">>" not in rxn_smiles:
            return False
            
        try:
            rxn_parts = rxn_smiles.split(">>")
            reactants_smiles = rxn_parts[0]
            products_smiles = rxn_parts[1]
            
            # Parse products
            product_mols = []
            for prod_smi in products_smiles.split("."):
                mol = Chem.MolFromSmiles(prod_smi)
                if mol:
                    product_mols.append(mol)
            
            # Parse reactants  
            reactant_mols = []
            for react_smi in reactants_smiles.split("."):
                mol = Chem.MolFromSmiles(react_smi)
                if mol:
                    reactant_mols.append(mol)
            
            # Check if ring pattern is present in products
            ring_in_products = any(mol.HasSubstructMatch(self.ring_pattern) for mol in product_mols)
            
            if not ring_in_products:
                return False
                
            # Check if ring pattern is absent in reactants (indicating ring formation)
            ring_in_reactants = any(mol.HasSubstructMatch(self.ring_pattern) for mol in reactant_mols)
            
            # Ring formation occurs if ring is in products but not in reactants
            return ring_in_products and not ring_in_reactants
            
        except Exception:
            return False
