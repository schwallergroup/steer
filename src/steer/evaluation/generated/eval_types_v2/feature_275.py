"""Generated evaluation code for: Protecting group swap Cbz to Boc"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ProtectingGroupSwap(MultiRxnCondBase):
    """
    Evaluates routes for protecting group swap from Cbz to Boc on nitrogen.
    Checks for Cbz deprotection immediately followed by Boc protection.
    """
    
    def __init__(self, config):
        self.protection_sequence = config.get("protection_sequence", ["cbz_deprotection", "boc_protection"])
        self.consecutive = config.get("consecutive", True)
        self.protecting_atom = config.get("protecting_atom", "N")
        
        # SMARTS patterns for protecting groups
        self.cbz_pattern = "[NX3][C](=O)O[CH2]c1ccccc1"  # Cbz-protected nitrogen
        self.boc_pattern = "[NX3][C](=O)O[C]([CH3])([CH3])[CH3]"  # Boc-protected nitrogen
        self.free_amine_pattern = "[NX3H2,NX3H1]"  # Free primary or secondary amine
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        if len(reactions) < 2:
            return False, len(reactions)
        
        # Find Cbz deprotection and Boc protection reactions
        cbz_deprotection_indices = []
        boc_protection_indices = []
        
        for i, rxn in enumerate(reactions):
            if self.is_cbz_deprotection(rxn):
                cbz_deprotection_indices.append(i)
            if self.is_boc_protection(rxn):
                boc_protection_indices.append(i)
        
        # Check if we have both types of reactions
        if not cbz_deprotection_indices or not boc_protection_indices:
            return False, len(reactions)
        
        # If consecutive is required, check for adjacent reactions
        if self.consecutive:
            for cbz_idx in cbz_deprotection_indices:
                for boc_idx in boc_protection_indices:
                    if abs(cbz_idx - boc_idx) == 1:
                        # Verify the sequence order (Cbz deprotection should come before Boc protection)
                        if cbz_idx < boc_idx:
                            return True, len(reactions)
        else:
            # Just check that both occur in the route
            return True, len(reactions)
        
        return False, len(reactions)
    
    def is_cbz_deprotection(self, rxn):
        """Check if reaction is Cbz deprotection (Cbz-N -> free amine)"""
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[0].split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in rxn_parts[1].split(".")]
            
            if not all(reactants) or not all(products):
                return False
            
            # Check for Cbz pattern in reactants and free amine in products
            has_cbz_reactant = any(mol.HasSubstructMatch(Chem.MolFromSmarts(self.cbz_pattern)) 
                                  for mol in reactants)
            has_free_amine_product = any(mol.HasSubstructMatch(Chem.MolFromSmarts(self.free_amine_pattern)) 
                                        for mol in products)
            
            # Should not have Cbz pattern in products
            has_cbz_product = any(mol.HasSubstructMatch(Chem.MolFromSmarts(self.cbz_pattern)) 
                                 for mol in products)
            
            return has_cbz_reactant and has_free_amine_product and not has_cbz_product
            
        except Exception:
            return False
    
    def is_boc_protection(self, rxn):
        """Check if reaction is Boc protection (free amine -> Boc-N)"""
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[0].split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in rxn_parts[1].split(".")]
            
            if not all(reactants) or not all(products):
                return False
            
            # Check for free amine in reactants and Boc pattern in products
            has_free_amine_reactant = any(mol.HasSubstructMatch(Chem.MolFromSmarts(self.free_amine_pattern)) 
                                         for mol in reactants)
            has_boc_product = any(mol.HasSubstructMatch(Chem.MolFromSmarts(self.boc_pattern)) 
                                 for mol in products)
            
            # Should not have Boc pattern in reactants
            has_boc_reactant = any(mol.HasSubstructMatch(Chem.MolFromSmarts(self.boc_pattern)) 
                                  for mol in reactants)
            
            return has_free_amine_reactant and has_boc_product and not has_boc_reactant
            
        except Exception:
            return False
