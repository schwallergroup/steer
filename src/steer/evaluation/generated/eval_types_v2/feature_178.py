"""Generated evaluation code for: Benzyl ether followed by deprotection strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BenzylEtherProtectDeprotect(MultiRxnCondBase):
    """
    Evaluates routes that use benzyl protection of phenol followed by deprotection.
    Detects protection-deprotection cycles involving benzyl ethers.
    """
    
    def __init__(self, config):
        self.require_cycle = config.get("require_cycle", True)
        self.max_cycle_length = config.get("max_cycle_length", 5)
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Find benzyl protection and deprotection reactions
        protection_indices = []
        deprotection_indices = []
        
        for i, rxn in enumerate(reactions):
            if self.detect_benzyl_protection(rxn):
                protection_indices.append(i)
            elif self.detect_benzyl_deprotection(rxn):
                deprotection_indices.append(i)
        
        # Check for protection-deprotection cycle
        has_cycle = self.has_protect_deprotect_cycle(
            protection_indices, deprotection_indices, len(reactions)
        )
        
        condition_met = has_cycle if self.require_cycle else (
            len(protection_indices) > 0 or len(deprotection_indices) > 0
        )
        
        return condition_met, len(reactions)
    
    def detect_benzyl_protection(self, rxn):
        """Detect benzyl protection of phenol: phenol + benzyl halide -> benzyl ether"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0].split(".")
        products = rxn_parts[1].split(".")
        
        # Look for phenol in reactants and benzyl ether in products
        has_phenol = False
        has_benzyl_reagent = False
        has_benzyl_ether = False
        
        phenol_pattern = Chem.MolFromSmarts("[OH1][c]")
        benzyl_halide_pattern = Chem.MolFromSmarts("[CH2][c]1[cH][cH][cH][cH][cH]1")
        benzyl_ether_pattern = Chem.MolFromSmarts("[CH2][c]1[cH][cH][cH][cH][cH]1-[CH2]-O-[c]")
        
        for reactant in reactants:
            mol = Chem.MolFromSmiles(reactant)
            if mol and mol.HasSubstructMatch(phenol_pattern):
                has_phenol = True
            if mol and mol.HasSubstructMatch(benzyl_halide_pattern):
                has_benzyl_reagent = True
                
        for product in products:
            mol = Chem.MolFromSmiles(product)
            if mol and mol.HasSubstructMatch(benzyl_ether_pattern):
                has_benzyl_ether = True
        
        return has_phenol and has_benzyl_reagent and has_benzyl_ether
    
    def detect_benzyl_deprotection(self, rxn):
        """Detect benzyl deprotection: benzyl ether -> phenol"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0].split(".")
        products = rxn_parts[1].split(".")
        
        has_benzyl_ether = False
        has_phenol = False
        
        benzyl_ether_pattern = Chem.MolFromSmarts("[CH2][c]1[cH][cH][cH][cH][cH]1-[CH2]-O-[c]")
        phenol_pattern = Chem.MolFromSmarts("[OH1][c]")
        
        for reactant in reactants:
            mol = Chem.MolFromSmiles(reactant)
            if mol and mol.HasSubstructMatch(benzyl_ether_pattern):
                has_benzyl_ether = True
                
        for product in products:
            mol = Chem.MolFromSmiles(product)
            if mol and mol.HasSubstructMatch(phenol_pattern):
                has_phenol = True
        
        return has_benzyl_ether and has_phenol
    
    def has_protect_deprotect_cycle(self, protection_indices, deprotection_indices, total_reactions):
        """Check if protection is followed by deprotection within max_cycle_length steps"""
        if not protection_indices or not deprotection_indices:
            return False
            
        for prot_idx in protection_indices:
            for deprot_idx in deprotection_indices:
                # Deprotection should come after protection
                if deprot_idx > prot_idx:
                    cycle_length = deprot_idx - prot_idx
                    if cycle_length <= self.max_cycle_length:
                        return True
        
        return False
    
    def route_scoring(self, x):
        """Score based on presence of protection-deprotection cycle"""
        if x < 0:
            return 0  # No cycle found
        else:
            return 10 - x  # Earlier cycles are better (less synthetic steps wasted)
