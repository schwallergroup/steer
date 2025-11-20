"""Generated evaluation code for: Trityl protecting group for primary alcohol"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class TritylProtectingGroup(BaseScoring):
    """
    Evaluates routes that use trityl protecting group strategy for primary alcohols.
    Checks for trityl protection/deprotection reactions and scores based on deprotection depth.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "depth")
        self.target_depth = config.get("target_depth", {}).get("value", 0.2)
        
        # SMARTS patterns for detection
        self.trityl_group = Chem.MolFromSmarts("[CH2][O]C(c1ccccc1)(c2ccccc2)c3ccccc3")
        self.primary_alcohol = Chem.MolFromSmarts("[CH2][OH]")
        self.trityl_cation = Chem.MolFromSmarts("C(c1ccccc1)(c2ccccc2)c3ccccc3")

    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Protection strategy not found
        
        if self.condition_type == "bool":
            return 1 if x >= 0 else 0
        else:
            # Late-stage deprotection is preferred (lower depth fraction)
            return max(0, 1 - x)

    def hit_condition(self, d):
        """
        Detects trityl deprotection reactions by checking for:
        1. Trityl-protected alcohol in reactants
        2. Free primary alcohol in products
        3. Trityl cation or related leaving group
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
                
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse reactants and products
            reactants = []
            for r_smiles in reactants_smiles.split("."):
                mol = Chem.MolFromSmiles(r_smiles)
                if mol:
                    reactants.append(mol)
                    
            products = []
            for p_smiles in products_smiles.split("."):
                mol = Chem.MolFromSmiles(p_smiles)
                if mol:
                    products.append(mol)
            
            if not reactants or not products:
                return False
            
            # Check for trityl deprotection pattern
            has_trityl_reactant = any(mol.HasSubstructMatch(self.trityl_group) for mol in reactants)
            has_alcohol_product = any(mol.HasSubstructMatch(self.primary_alcohol) for mol in products)
            has_trityl_leaving = any(mol.HasSubstructMatch(self.trityl_cation) for mol in products)
            
            # Alternative check: look for the reverse (protection reaction)
            has_alcohol_reactant = any(mol.HasSubstructMatch(self.primary_alcohol) for mol in reactants)
            has_trityl_product = any(mol.HasSubstructMatch(self.trityl_group) for mol in products)
            
            # Return True if either protection or deprotection is detected
            deprotection = has_trityl_reactant and has_alcohol_product
            protection = has_alcohol_reactant and has_trityl_product
            
            return deprotection or protection
            
        except Exception:
            return False
