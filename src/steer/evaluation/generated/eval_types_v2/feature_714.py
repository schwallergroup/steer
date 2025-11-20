"""Generated evaluation code for: Benzyl protecting group strategy for phenol"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BenzylPhenolProtection(BaseScoring):
    """
    Evaluates synthesis routes for benzyl protecting group strategy on phenols.
    Checks if a benzyl ether protection is formed and deprotected late in the synthesis.
    """
    
    def __init__(self, config: Dict):
        self.deprotection_timing = config["parameters"].get("deprotection_timing", "late")
        # SMARTS pattern for benzyl ether (Ar-O-CH2-Ph)
        self.benzyl_ether_pattern = "[OH1][CH2]c1ccccc1"
        # SMARTS pattern for free phenol
        self.phenol_pattern = "c[OH1]"
        
    def route_scoring(self, x) -> float:
        """
        Score based on deprotection depth:
        - Late deprotection (high depth fraction) gets higher score
        - Early deprotection gets lower score
        """
        if x < 0:
            return 0  # Protection/deprotection strategy not found
        
        if self.deprotection_timing == "late":
            return 1 - x  # Reward late deprotection (high depth fraction)
        else:
            return x  # Reward early deprotection (low depth fraction)
    
    def hit_condition(self, d):
        """
        Check if this reaction involves benzyl ether deprotection of a phenol.
        Look for benzyl ether in reactants and free phenol in products.
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse reactants and products
            reactants = []
            for r_smi in reactants_smiles.split("."):
                mol = Chem.MolFromSmiles(r_smi)
                if mol is not None:
                    reactants.append(mol)
            
            products = []
            for p_smi in products_smiles.split("."):
                mol = Chem.MolFromSmiles(p_smi)
                if mol is not None:
                    products.append(mol)
            
            if not reactants or not products:
                return False
            
            # Check for benzyl ether in reactants
            benzyl_ether_mol = Chem.MolFromSmarts(self.benzyl_ether_pattern)
            has_benzyl_ether_reactant = any(
                mol.HasSubstructMatch(benzyl_ether_mol) for mol in reactants
            )
            
            # Check for free phenol in products
            phenol_mol = Chem.MolFromSmarts(self.phenol_pattern)
            has_phenol_product = any(
                mol.HasSubstructMatch(phenol_mol) for mol in products
            )
            
            # This is a benzyl deprotection if we have benzyl ether -> phenol
            return has_benzyl_ether_reactant and has_phenol_product
            
        except Exception:
            return False
