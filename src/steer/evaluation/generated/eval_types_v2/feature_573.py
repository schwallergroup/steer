"""Generated evaluation code for: Carboxylic acid protection deprotection cycling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class CarboxylicAcidProtectionCycling(MultiRxnCondBase):
    """
    Evaluates routes for carboxylic acid protection-deprotection cycling using ethyl esters.
    Detects esterification followed by hydrolysis cycles of carboxylic acid groups.
    """
    
    def __init__(self, config):
        self.cycle_count = config.get("cycle_count", 1)
        self.carboxylic_acid_pattern = Chem.MolFromSmarts("C(=O)[OH]")
        self.ethyl_ester_pattern = Chem.MolFromSmarts("C(=O)OCC")
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Track protection and deprotection events
        protection_events = []
        deprotection_events = []
        
        for i, rxn in enumerate(reactions):
            if self.detect_esterification(rxn):
                protection_events.append(i)
            elif self.detect_hydrolysis(rxn):
                deprotection_events.append(i)
        
        # Count complete cycles (protection followed by deprotection)
        cycles_found = self.count_protection_cycles(protection_events, deprotection_events)
        
        condition = cycles_found >= self.cycle_count
        return condition, len(reactions)
    
    def detect_esterification(self, rxn):
        """Detect carboxylic acid -> ethyl ester conversion"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[0].split(".") if r.strip()]
        products = [Chem.MolFromSmiles(p.strip()) for p in rxn_parts[1].split(".") if p.strip()]
        
        if not all(reactants) or not all(products):
            return False
        
        # Check for carboxylic acid in reactants and ethyl ester in products
        has_carboxylic_acid_reactant = any(mol.HasSubstructMatch(self.carboxylic_acid_pattern) for mol in reactants)
        has_ethyl_ester_product = any(mol.HasSubstructMatch(self.ethyl_ester_pattern) for mol in products)
        
        return has_carboxylic_acid_reactant and has_ethyl_ester_product
    
    def detect_hydrolysis(self, rxn):
        """Detect ethyl ester -> carboxylic acid conversion"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[0].split(".") if r.strip()]
        products = [Chem.MolFromSmiles(p.strip()) for p in rxn_parts[1].split(".") if p.strip()]
        
        if not all(reactants) or not all(products):
            return False
        
        # Check for ethyl ester in reactants and carboxylic acid in products
        has_ethyl_ester_reactant = any(mol.HasSubstructMatch(self.ethyl_ester_pattern) for mol in reactants)
        has_carboxylic_acid_product = any(mol.HasSubstructMatch(self.carboxylic_acid_pattern) for mol in products)
        
        return has_ethyl_ester_reactant and has_carboxylic_acid_product
    
    def count_protection_cycles(self, protection_events, deprotection_events):
        """Count complete protection-deprotection cycles"""
        if not protection_events or not deprotection_events:
            return 0
        
        cycles = 0
        used_deprotections = set()
        
        for prot_idx in protection_events:
            # Find the next deprotection event after this protection
            for deprot_idx in deprotection_events:
                if deprot_idx > prot_idx and deprot_idx not in used_deprotections:
                    cycles += 1
                    used_deprotections.add(deprot_idx)
                    break
        
        return cycles
