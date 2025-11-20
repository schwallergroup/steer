"""Generated evaluation code for: Cbz protecting group strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class CbzProtectionStrategy(MultiRxnCondBase):
    """
    Evaluates synthesis routes for proper Cbz protecting group strategy.
    Checks for protection of amine with Cbz followed by deprotection in a cycle.
    """
    
    def __init__(self, config):
        self.require_protection_deprotection_cycle = config.get("protection_deprotection_cycle", True)
        self.functional_group = config.get("functional_group", "amine")
        self.protecting_group = config.get("protecting_group", "Cbz")
        
        # SMARTS patterns for Cbz-protected amine and free amine
        self.cbz_protected_amine = "[NH1][CH2]c1ccccc1"  # N-benzyloxycarbonyl amine
        self.free_amine_pattern = "[NH2,NH1]"  # Primary or secondary amine
        self.cbz_leaving_pattern = "O=C(O[CH2]c1ccccc1)[NH]"  # Cbz carbamate
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Track Cbz protection and deprotection events
        protection_found = False
        deprotection_found = False
        protection_depth = -1
        deprotection_depth = -1
        
        for i, rxn in enumerate(reactions):
            if self.detect_cbz_protection(rxn):
                protection_found = True
                protection_depth = i
            elif self.detect_cbz_deprotection(rxn):
                deprotection_found = True
                deprotection_depth = i
        
        if self.require_protection_deprotection_cycle:
            # Both protection and deprotection must occur, with protection happening first
            condition = (protection_found and deprotection_found and 
                        protection_depth < deprotection_depth)
        else:
            # Just require either protection or deprotection
            condition = protection_found or deprotection_found
        
        # Return the depth where the strategy is completed (deprotection if cycle required)
        strategy_depth = deprotection_depth if (condition and deprotection_found) else protection_depth
        return condition, strategy_depth if strategy_depth >= 0 else len(reactions)
    
    def detect_cbz_protection(self, rxn):
        """Detect Cbz protection of amine: amine + Cbz-Cl -> Cbz-protected amine"""
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[0].split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in rxn_parts[1].split(".")]
            
            if not all(reactants) or not all(products):
                return False
            
            # Check if reactants contain free amine and products contain Cbz-protected amine
            has_free_amine_reactant = any(mol.HasSubstructMatch(Chem.MolFromSmarts(self.free_amine_pattern)) 
                                        for mol in reactants)
            has_cbz_product = any(mol.HasSubstructMatch(Chem.MolFromSmarts(self.cbz_protected_amine)) 
                                for mol in products)
            
            # Look for Cbz reagent (benzyl chloroformate or similar)
            cbz_reagent_pattern = "ClC(=O)O[CH2]c1ccccc1"  # Cbz-Cl
            has_cbz_reagent = any(mol.HasSubstructMatch(Chem.MolFromSmarts(cbz_reagent_pattern)) 
                                for mol in reactants)
            
            return has_free_amine_reactant and has_cbz_product and has_cbz_reagent
            
        except Exception:
            return False
    
    def detect_cbz_deprotection(self, rxn):
        """Detect Cbz deprotection: Cbz-protected amine -> free amine + byproducts"""
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[0].split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in rxn_parts[1].split(".")]
            
            if not all(reactants) or not all(products):
                return False
            
            # Check if reactants contain Cbz-protected amine and products contain free amine
            has_cbz_reactant = any(mol.HasSubstructMatch(Chem.MolFromSmarts(self.cbz_protected_amine)) 
                                 for mol in reactants)
            has_free_amine_product = any(mol.HasSubstructMatch(Chem.MolFromSmarts(self.free_amine_pattern)) 
                                       for mol in products)
            
            # Look for typical deprotection byproducts (toluene, CO2)
            toluene_pattern = "Cc1ccccc1"
            has_toluene_byproduct = any(mol.HasSubstructMatch(Chem.MolFromSmarts(toluene_pattern)) 
                                      for mol in products)
            
            return has_cbz_reactant and has_free_amine_product and has_toluene_byproduct
            
        except Exception:
            return False
