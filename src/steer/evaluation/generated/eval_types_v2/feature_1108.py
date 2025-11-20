"""Generated evaluation code for: Boc protecting group strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BocProtectingGroupStrategy(BaseScoring):
    """
    Evaluates synthesis routes based on Boc protecting group strategy for amines.
    Checks if Boc protection/deprotection occurs and at what depth in the route.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "bool")
        self.target_depth = config.get("target_depth", {}).get("value", -1)
        
        # Boc protecting group SMARTS patterns
        self.boc_pattern = Chem.MolFromSmarts("[NH1,NH2,N]C(=O)OC(C)(C)C")  # Boc-protected amine
        self.free_amine_pattern = Chem.MolFromSmarts("[NH1,NH2]")  # Free amine
        
    def route_scoring(self, x) -> float:
        """Convert depth fraction to score (0-10)"""
        if self.condition_type == "bool":
            if self.target_depth == -1:  # Positive if condition not met
                return 1 if x < 0 else 0
        else:
            if x < 0:
                return 0  # Strategy not used
            return max(0, 1 - abs(x - self.target_depth))
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction involves Boc protection/deprotection of amines"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            reactants_smiles, products_smiles = mapped_rxn.split(">>")
            
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
                
            # Check for Boc protection: free amine -> Boc-protected amine
            reactant_has_free_amine = any(mol.HasSubstructMatch(self.free_amine_pattern) for mol in reactants)
            product_has_boc = any(mol.HasSubstructMatch(self.boc_pattern) for mol in products)
            
            # Check for Boc deprotection: Boc-protected amine -> free amine
            reactant_has_boc = any(mol.HasSubstructMatch(self.boc_pattern) for mol in reactants)
            product_has_free_amine = any(mol.HasSubstructMatch(self.free_amine_pattern) for mol in products)
            
            # Return True if either Boc protection or deprotection occurs
            boc_protection = reactant_has_free_amine and product_has_boc
            boc_deprotection = reactant_has_boc and product_has_free_amine
            
            return boc_protection or boc_deprotection
            
        except Exception:
            return False
