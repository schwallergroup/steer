"""Generated evaluation code for: Boc protecting group strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BocProtectionStrategy(MultiRxnCondBase):
    """
    Evaluates synthesis routes for proper use of Boc (tert-butoxycarbonyl) protecting group strategy.
    Checks for Boc protection of amines followed by deprotection later in the route.
    """
    
    def __init__(self, config):
        self.protecting_group = config.get("protecting_group", "Boc")
        self.functional_group = config.get("functional_group", "amine")
        self.boc_smarts = "[NX3][C](=O)OC(C)(C)C"  # Boc-protected amine pattern
        self.free_amine_smarts = "[NX3H2,NX3H1]"  # Primary or secondary amine
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        has_protection = False
        has_deprotection = False
        protection_before_deprotection = False
        
        protection_step = -1
        deprotection_step = -1
        
        # Check reactions in order
        for i, rxn in enumerate(reactions):
            if self.detect_boc_protection(rxn):
                has_protection = True
                if protection_step == -1:
                    protection_step = i
            
            if self.detect_boc_deprotection(rxn):
                has_deprotection = True
                if deprotection_step == -1:
                    deprotection_step = i
        
        # Protection should occur before deprotection
        if has_protection and has_deprotection and protection_step < deprotection_step:
            protection_before_deprotection = True
        
        # Condition is met if we have proper Boc protection strategy
        condition = has_protection and has_deprotection and protection_before_deprotection
        
        return condition, len(reactions)
    
    def detect_boc_protection(self, rxn):
        """
        Detects Boc protection reaction: free amine -> Boc-protected amine
        """
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        try:
            # Check if reactants contain free amine and products contain Boc-protected amine
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".")]
            
            has_free_amine_reactant = any(
                mol and mol.HasSubstructMatch(Chem.MolFromSmarts(self.free_amine_smarts)) 
                for mol in reactant_mols
            )
            
            has_boc_product = any(
                mol and mol.HasSubstructMatch(Chem.MolFromSmarts(self.boc_smarts))
                for mol in product_mols
            )
            
            has_boc_reagent = any(
                mol and "BOC" in Chem.MolToSmiles(mol).upper() or 
                mol and mol.HasSubstructMatch(Chem.MolFromSmarts("CC(C)(C)OC(=O)Cl"))  # Boc-Cl
                for mol in reactant_mols if mol
            )
            
            return has_free_amine_reactant and has_boc_product and has_boc_reagent
            
        except:
            return False
    
    def detect_boc_deprotection(self, rxn):
        """
        Detects Boc deprotection reaction: Boc-protected amine -> free amine
        """
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        try:
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".")]
            
            has_boc_reactant = any(
                mol and mol.HasSubstructMatch(Chem.MolFromSmarts(self.boc_smarts))
                for mol in reactant_mols
            )
            
            has_free_amine_product = any(
                mol and mol.HasSubstructMatch(Chem.MolFromSmarts(self.free_amine_smarts))
                for mol in product_mols
            )
            
            # Common deprotection conditions (TFA, HCl, etc.)
            has_deprotection_reagent = any(
                mol and ("CF3" in Chem.MolToSmiles(mol) or  # TFA
                        "Cl" in Chem.MolToSmiles(mol))  # HCl
                for mol in reactant_mols if mol
            )
            
            return has_boc_reactant and has_free_amine_product
            
        except:
            return False
