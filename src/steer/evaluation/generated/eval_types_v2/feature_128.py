"""Generated evaluation code for: Multiple protecting group cycling strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ProtectingGroupStrategy(MultiRxnCondBase):
    """
    Evaluates synthesis routes based on multiple protecting group cycling strategies.
    Checks for specific protection/deprotection patterns and counts cycles.
    """
    
    def __init__(self, config):
        self.protection_steps = config.get("protection_steps", [])
        self.deprotection_steps = config.get("deprotection_steps", [])
        self.target_cycle_count = config.get("cycle_count", 1)
        
        # Define SMARTS patterns for protecting groups
        self.boc_pattern = "[N:1][C](=O)OC(C)(C)C"  # Boc-protected amine
        self.benzyl_pattern = "[N:1]Cc1ccccc1"  # Benzyl-protected amine
        self.free_amine_pattern = "[NH2:1]"  # Free amine
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Track protection/deprotection events
        boc_protections = 0
        boc_deprotections = 0
        benzyl_protections = 0
        benzyl_deprotections = 0
        
        for rxn in reactions:
            if self.detect_boc_protection(rxn):
                boc_protections += 1
            elif self.detect_boc_deprotection(rxn):
                boc_deprotections += 1
            elif self.detect_benzyl_protection(rxn):
                benzyl_protections += 1
            elif self.detect_benzyl_deprotection(rxn):
                benzyl_deprotections += 1
        
        # Calculate cycles (minimum of protection and deprotection events)
        boc_cycles = min(boc_protections, boc_deprotections)
        benzyl_cycles = min(benzyl_protections, benzyl_deprotections)
        total_cycles = boc_cycles + benzyl_cycles
        
        # Check if strategy matches expected pattern
        has_boc_strategy = "Boc protection" in self.protection_steps and "Boc deprotection" in self.deprotection_steps
        has_benzyl_strategy = any("benzyl protection" in step for step in self.protection_steps) and "dibenzyl deprotection" in self.deprotection_steps
        
        strategy_matches = True
        if has_boc_strategy and boc_cycles == 0:
            strategy_matches = False
        if has_benzyl_strategy and benzyl_cycles == 0:
            strategy_matches = False
        
        # Condition met if we have the right strategy and cycle count
        condition = strategy_matches and total_cycles >= self.target_cycle_count
        
        return condition, len(reactions)
    
    def detect_boc_protection(self, rxn):
        """Detect Boc protection: free amine -> Boc-protected amine"""
        return self._detect_protection_reaction(rxn, self.free_amine_pattern, self.boc_pattern)
    
    def detect_boc_deprotection(self, rxn):
        """Detect Boc deprotection: Boc-protected amine -> free amine"""
        return self._detect_protection_reaction(rxn, self.boc_pattern, self.free_amine_pattern)
    
    def detect_benzyl_protection(self, rxn):
        """Detect benzyl protection: free amine -> benzyl-protected amine"""
        return self._detect_protection_reaction(rxn, self.free_amine_pattern, self.benzyl_pattern)
    
    def detect_benzyl_deprotection(self, rxn):
        """Detect benzyl deprotection: benzyl-protected amine -> free amine"""
        return self._detect_protection_reaction(rxn, self.benzyl_pattern, self.free_amine_pattern)
    
    def _detect_protection_reaction(self, rxn, reactant_pattern, product_pattern):
        """
        Generic method to detect protection/deprotection reactions
        by comparing substructure patterns in reactants vs products
        """
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[0].split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in rxn_parts[1].split(".")]
            
            # Remove None molecules (failed parsing)
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            if not reactants or not products:
                return False
            
            # Create pattern molecules
            reactant_mol_pattern = Chem.MolFromSmarts(reactant_pattern)
            product_mol_pattern = Chem.MolFromSmarts(product_pattern)
            
            if reactant_mol_pattern is None or product_mol_pattern is None:
                return False
            
            # Check if reactant pattern exists in reactants
            has_reactant_pattern = any(mol.HasSubstructMatch(reactant_mol_pattern) for mol in reactants)
            
            # Check if product pattern exists in products  
            has_product_pattern = any(mol.HasSubstructMatch(product_mol_pattern) for mol in products)
            
            return has_reactant_pattern and has_product_pattern
            
        except Exception:
            return False
