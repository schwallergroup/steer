"""Generated evaluation code for: Sequential protecting group swap TFA to Boc"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SequentialProtectingGroupSwap(BaseScoring):
    """
    Evaluates routes for sequential protecting group swap from TFA to Boc.
    Checks for deprotection of trifluoroacetamide followed by Boc protection
    within the specified step range.
    """
    
    def __init__(self, config: Dict):
        self.initial_pg = config["parameters"]["initial_pg"]
        self.final_pg = config["parameters"]["final_pg"]
        self.deprotection_step = config["parameters"]["deprotection_step"]
        self.protection_step = config["parameters"]["protection_step"]
        
        # SMARTS patterns for protecting groups
        self.tfa_pattern = "NC(=O)C(F)(F)F"  # Trifluoroacetamide
        self.boc_pattern = "NC(=O)OC(C)(C)C"  # Boc carbamate
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Sequential swap doesn't occur
        else:
            # Better score for earlier occurrence of the swap
            return max(0, 10 - x * 8)
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction is part of a sequential TFA to Boc swap"""
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
                
            reactants, products = rxn_smiles.split(">>")
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".")]
            
            # Filter out None molecules
            reactant_mols = [mol for mol in reactant_mols if mol is not None]
            product_mols = [mol for mol in product_mols if mol is not None]
            
            # Check for TFA deprotection (TFA group disappears)
            tfa_in_reactants = any(mol.HasSubstructMatch(Chem.MolFromSmarts(self.tfa_pattern)) 
                                 for mol in reactant_mols)
            tfa_in_products = any(mol.HasSubstructMatch(Chem.MolFromSmarts(self.tfa_pattern)) 
                                for mol in product_mols)
            
            # Check for Boc protection (Boc group appears)
            boc_in_reactants = any(mol.HasSubstructMatch(Chem.MolFromSmarts(self.boc_pattern)) 
                                 for mol in reactant_mols)
            boc_in_products = any(mol.HasSubstructMatch(Chem.MolFromSmarts(self.boc_pattern)) 
                                for mol in product_mols)
            
            # Sequential swap: either deprotection step or protection step
            is_deprotection = tfa_in_reactants and not tfa_in_products
            is_protection = not boc_in_reactants and boc_in_products
            
            # Check step constraints if provided
            current_step = d.get("depth", 0)
            
            if is_deprotection and self.deprotection_step > 0:
                return abs(current_step - self.deprotection_step) <= 1
            elif is_protection and self.protection_step > 0:
                return abs(current_step - self.protection_step) <= 1
            
            # If no step constraints, accept either deprotection or protection
            return is_deprotection or is_protection
            
        except Exception:
            return False
