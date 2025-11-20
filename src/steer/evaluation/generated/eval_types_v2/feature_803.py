"""Generated evaluation code for: Early lactam ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlyLactamRingFormation(BaseScoring):
    """
    Evaluates if a lactam ring formation occurs early in the synthesis route.
    Detects the formation of a five-membered lactam ring via intramolecular cyclization
    and scores based on how early this occurs relative to the target depth.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]
        self.formation_depth = config["parameters"]["formation_depth"]
        self.lactam_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        
        if self.timing == "early":
            # Early formation is better - penalize late formation
            if x <= self.formation_depth / 10.0:  # x is depth fraction
                return 1.0  # Perfect score for very early formation
            else:
                return max(0, 1.0 - (x - self.formation_depth / 10.0) * 2)
        else:
            # For other timing preferences
            return 1.0 - abs(x - self.formation_depth / 10.0)
    
    def hit_condition(self, d):
        """
        Checks if the reaction involves lactam ring formation by comparing
        reactants and products for the presence of the lactam pattern.
        """
        rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
        if not rxn_smiles or ">>" not in rxn_smiles:
            return False
            
        try:
            rxn_parts = rxn_smiles.split(">>")
            reactants_smiles = rxn_parts[0]
            products_smiles = rxn_parts[1]
            
            # Parse reactants
            reactant_mols = []
            for r_smiles in reactants_smiles.split("."):
                mol = Chem.MolFromSmiles(r_smiles)
                if mol is not None:
                    reactant_mols.append(mol)
            
            # Parse products  
            product_mols = []
            for p_smiles in products_smiles.split("."):
                mol = Chem.MolFromSmiles(p_smiles)
                if mol is not None:
                    product_mols.append(mol)
            
            # Check if lactam pattern is absent in reactants but present in products
            lactam_in_reactants = any(mol.HasSubstructMatch(self.lactam_pattern) 
                                    for mol in reactant_mols)
            lactam_in_products = any(mol.HasSubstructMatch(self.lactam_pattern) 
                                   for mol in product_mols)
            
            # Ring formation: lactam not in reactants but appears in products
            return not lactam_in_reactants and lactam_in_products
            
        except Exception:
            return False
