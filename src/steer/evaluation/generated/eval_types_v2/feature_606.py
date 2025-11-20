"""Generated evaluation code for: Cbz protection-deprotection cycle strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class CbzProtectionDeprotectionCycle(MultiRxnCondBase):
    """
    Evaluates whether a synthesis route includes a Cbz protection-deprotection cycle.
    Checks for the presence of both Cbz protection and deprotection reactions on amine groups.
    """
    
    def __init__(self, config):
        self.cycle_present = config.get("cycle_present", True)
        self.protecting_group = config.get("protecting_group", "Cbz")
        self.functional_group = config.get("functional_group", "amine")
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        has_protection = any(self.detect_cbz_protection(r) for r in reactions)
        has_deprotection = any(self.detect_cbz_deprotection(r) for r in reactions)
        
        cycle_detected = has_protection and has_deprotection
        condition = cycle_detected == self.cycle_present
        
        return condition, len(reactions)
    
    def detect_cbz_protection(self, rxn):
        """Detect Cbz protection of amine groups"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0].split(".")
        products = rxn_parts[1].split(".")
        
        # Look for formation of Cbz-protected amine (carbamate linkage)
        cbz_carbamate_pattern = "[NH1]C(=O)OCC1=CC=CC=C1"  # N-Cbz carbamate
        free_amine_pattern = "[NH2,NH1]"
        
        try:
            # Check if reactants have free amine and products have Cbz-protected amine
            has_free_amine_reactant = any(
                Chem.MolFromSmiles(r) and 
                Chem.MolFromSmiles(r).HasSubstructMatch(Chem.MolFromSmarts(free_amine_pattern))
                for r in reactants
            )
            
            has_cbz_product = any(
                Chem.MolFromSmiles(p) and 
                Chem.MolFromSmiles(p).HasSubstructMatch(Chem.MolFromSmarts(cbz_carbamate_pattern))
                for p in products
            )
            
            # Also check for Cbz-Cl reagent in reactants
            cbz_cl_pattern = "ClC(=O)OCC1=CC=CC=C1"
            has_cbz_reagent = any(
                Chem.MolFromSmiles(r) and 
                Chem.MolFromSmiles(r).HasSubstructMatch(Chem.MolFromSmarts(cbz_cl_pattern))
                for r in reactants
            )
            
            return has_free_amine_reactant and has_cbz_product and has_cbz_reagent
            
        except:
            return False
    
    def detect_cbz_deprotection(self, rxn):
        """Detect Cbz deprotection to regenerate free amine"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0].split(".")
        products = rxn_parts[1].split(".")
        
        # Look for cleavage of Cbz-protected amine
        cbz_carbamate_pattern = "[NH1]C(=O)OCC1=CC=CC=C1"  # N-Cbz carbamate
        free_amine_pattern = "[NH2,NH1]"
        
        try:
            # Check if reactants have Cbz-protected amine and products have free amine
            has_cbz_reactant = any(
                Chem.MolFromSmiles(r) and 
                Chem.MolFromSmiles(r).HasSubstructMatch(Chem.MolFromSmarts(cbz_carbamate_pattern))
                for r in reactants
            )
            
            has_free_amine_product = any(
                Chem.MolFromSmiles(p) and 
                Chem.MolFromSmiles(p).HasSubstructMatch(Chem.MolFromSmarts(free_amine_pattern))
                for p in products
            )
            
            # Common deprotection byproduct: benzyl alcohol or CO2 + toluene
            deprotection_byproduct = any(
                Chem.MolFromSmiles(p) and (
                    Chem.MolFromSmiles(p).HasSubstructMatch(Chem.MolFromSmarts("OCC1=CC=CC=C1")) or  # benzyl alcohol
                    Chem.MolFromSmiles(p).HasSubstructMatch(Chem.MolFromSmarts("CC1=CC=CC=C1"))      # toluene
                )
                for p in products
            )
            
            return has_cbz_reactant and has_free_amine_product and deprotection_byproduct
            
        except:
            return False
