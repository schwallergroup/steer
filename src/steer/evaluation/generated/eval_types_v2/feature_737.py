"""Generated evaluation code for: PMB protecting group strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class PMBProtectingGroupStrategy(BaseScoring):
    """
    Evaluates PMB (para-methoxybenzyl) protecting group strategy for amines.
    Checks for PMB protection followed by late-stage deprotection.
    """
    
    def __init__(self, config: Dict):
        self.protecting_group = config["parameters"]["protecting_group"]
        self.functional_group = config["parameters"]["functional_group"]
        self.timing = config["parameters"]["timing"]
        
        # PMB pattern: para-methoxybenzyl group
        self.pmb_pattern = Chem.MolFromSmarts("[CH2]c1ccc(OC)cc1")
        # N-PMB protected amine pattern
        self.pmb_protected_amine = Chem.MolFromSmarts("N[CH2]c1ccc(OC)cc1")
        # Free amine pattern
        self.free_amine = Chem.MolFromSmarts("[NH2,NH1]")
    
    def route_scoring(self, x) -> float:
        """Score based on deprotection timing - later is better for late_stage_deprotection"""
        if x < 0:
            return 0  # No PMB deprotection found
        
        if self.timing == "late_stage_deprotection":
            # Reward later deprotection (closer to final product)
            return (1 - x) * 10
        else:
            # For other timing preferences, could implement different scoring
            return (1 - x) * 10
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction involves PMB deprotection of an amine"""
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
            
            # Check if reactants contain PMB-protected amine
            has_pmb_protected = any(
                mol.HasSubstructMatch(self.pmb_protected_amine) 
                for mol in reactants
            )
            
            # Check if products contain free amine (indicating deprotection)
            has_free_amine = any(
                mol.HasSubstructMatch(self.free_amine)
                for mol in products
            )
            
            # Check if PMB group is released as a byproduct or consumed
            pmb_consumed = any(
                mol.HasSubstructMatch(self.pmb_pattern)
                for mol in reactants
            ) and not any(
                mol.HasSubstructMatch(self.pmb_pattern)
                for mol in products if not self._is_pmb_byproduct(mol)
            )
            
            return has_pmb_protected and has_free_amine and pmb_consumed
            
        except Exception:
            return False
    
    def _is_pmb_byproduct(self, mol):
        """Check if molecule is a PMB-related byproduct (e.g., PMB alcohol, aldehyde)"""
        # Common PMB deprotection byproducts
        pmb_alcohol = Chem.MolFromSmarts("OCc1ccc(OC)cc1")  # PMB alcohol
        pmb_aldehyde = Chem.MolFromSmarts("O=Cc1ccc(OC)cc1")  # PMB aldehyde
        
        return (mol.HasSubstructMatch(pmb_alcohol) or 
                mol.HasSubstructMatch(pmb_aldehyde))
