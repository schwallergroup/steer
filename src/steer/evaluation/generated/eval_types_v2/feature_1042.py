"""Generated evaluation code for: Phthalimide protecting group strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class PhthalimideProtectingGroup(BaseScoring):
    """
    Evaluates synthesis routes for phthalimide protecting group strategy.
    Detects the use of phthalimide protection for amines followed by hydrazine deprotection
    as part of the Gabriel amine synthesis approach.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "fraction")
        self.target_depth = config.get("target_depth", {}).get("value", 0.5)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Strategy not found
        else:
            # Earlier use of protecting group strategy is generally better
            if self.condition_type == "bool":
                return 1  # Found the strategy
            else:
                return 1 - x  # Earlier depth gets higher score
    
    def hit_condition(self, d) -> bool:
        """
        Checks if a reaction involves phthalimide protecting group chemistry.
        This includes both protection (formation of phthalimide) and 
        deprotection (hydrazine cleavage) steps.
        """
        rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
        if not rxn_smiles or ">>" not in rxn_smiles:
            return False
        
        reactants_smiles, products_smiles = rxn_smiles.split(">>")
        
        # Parse molecules
        try:
            reactants = [Chem.MolFromSmiles(s.strip()) for s in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(s.strip()) for s in products_smiles.split(".")]
            
            if None in reactants or None in products:
                return False
        except:
            return False
        
        # Check for phthalimide protection step
        if self._is_phthalimide_protection(reactants, products):
            return True
        
        # Check for phthalimide deprotection step  
        if self._is_phthalimide_deprotection(reactants, products):
            return True
        
        return False
    
    def _is_phthalimide_protection(self, reactants, products) -> bool:
        """Check if reaction forms phthalimide from amine and phthalic anhydride/derivative"""
        # Phthalimide pattern - benzene ring fused to imide
        phthalimide_pattern = Chem.MolFromSmarts("c1cccc2c1C(=O)NC2=O")
        if not phthalimide_pattern:
            return False
        
        # Phthalic anhydride or N-hydroxyphthalimide as reactant
        phthalic_anhydride = Chem.MolFromSmarts("c1cccc2c1C(=O)OC2=O")
        phthalimide_derivative = Chem.MolFromSmarts("c1cccc2c1C(=O)N([OH,Br,Cl])C2=O")
        
        # Check if we have phthalic derivative in reactants and phthalimide in products
        has_phthalic_reactant = any(
            mol.HasSubstructMatch(phthalic_anhydride) or 
            mol.HasSubstructMatch(phthalimide_derivative) 
            for mol in reactants if mol
        )
        
        has_phthalimide_product = any(
            mol.HasSubstructMatch(phthalimide_pattern) 
            for mol in products if mol
        )
        
        return has_phthalic_reactant and has_phthalimide_product
    
    def _is_phthalimide_deprotection(self, reactants, products) -> bool:
        """Check if reaction cleaves phthalimide to release amine (Gabriel synthesis)"""
        # Phthalimide pattern in reactants
        phthalimide_pattern = Chem.MolFromSmarts("c1cccc2c1C(=O)NC2=O")
        if not phthalimide_pattern:
            return False
        
        # Hydrazine or similar nucleophile
        hydrazine_pattern = Chem.MolFromSmarts("NN")
        
        has_phthalimide_reactant = any(
            mol.HasSubstructMatch(phthalimide_pattern) 
            for mol in reactants if mol
        )
        
        has_hydrazine_reactant = any(
            mol.HasSubstructMatch(hydrazine_pattern) 
            for mol in reactants if mol
        )
        
        # Check if we get primary amine in products
        primary_amine_pattern = Chem.MolFromSmarts("[NH2;!$(NC=O)]")
        has_primary_amine_product = any(
            mol.HasSubstructMatch(primary_amine_pattern) 
            for mol in products if mol
        )
        
        return (has_phthalimide_reactant and 
                (has_hydrazine_reactant or has_primary_amine_product))
