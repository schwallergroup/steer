"""Generated evaluation code for: TBDPS protecting group strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class TBDPSProtectingGroupStrategy(BaseScoring):
    """
    Evaluates TBDPS protecting group strategy for alcohols.
    Checks if TBDPS protection occurs at appropriate depth in the synthesis route.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "bool")
        self.target_depth = config.get("target_depth", {}).get("value", -1)
        
        # TBDPS protecting group pattern: tert-butyldiphenylsilyl
        self.tbdps_pattern = Chem.MolFromSmarts("[Si]([C](C)(C)C)(c1ccccc1)(c2ccccc2)[O]")
        
        # Primary alcohol pattern
        self.primary_alcohol_pattern = Chem.MolFromSmarts("[CH2][OH]")
        
        # TBDPS-protected alcohol pattern
        self.tbdps_protected_pattern = Chem.MolFromSmarts("[CH2][O][Si]([C](C)(C)C)(c1ccccc1)(c2ccccc2)")
    
    def route_scoring(self, x) -> float:
        """Convert depth fraction to score (0-10)"""
        if self.condition_type == "bool":
            if self.target_depth == -1:  # Positive if condition not met
                return 1 if x < 0 else 0
        else:
            if x < 0:
                return 0  # Protection doesn't happen
            # Earlier protection is generally better for protecting group strategy
            return 1 - x
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves TBDPS protection of an alcohol.
        Returns True if TBDPS protecting group is introduced.
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
            
            # Check if TBDPS group appears in products but not in reactants
            tbdps_in_reactants = any(mol.HasSubstructMatch(self.tbdps_pattern) for mol in reactants)
            tbdps_in_products = any(mol.HasSubstructMatch(self.tbdps_pattern) for mol in products)
            
            # Check if there's a primary alcohol in reactants that gets protected
            alcohol_in_reactants = any(mol.HasSubstructMatch(self.primary_alcohol_pattern) for mol in reactants)
            protected_in_products = any(mol.HasSubstructMatch(self.tbdps_protected_pattern) for mol in products)
            
            # TBDPS protection occurs if:
            # 1. TBDPS appears in products but not reactants (new protection)
            # 2. There's an alcohol in reactants and protected alcohol in products
            return (not tbdps_in_reactants and tbdps_in_products and 
                    alcohol_in_reactants and protected_in_products)
                    
        except Exception:
            return False
