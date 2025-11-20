"""Generated evaluation code for: Sequential protecting group manipulations"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SequentialProtectingGroupStrategy(MultiRxnCondBase):
    """
    Evaluates sequential protecting group manipulations in synthesis routes.
    Checks if multiple protecting groups (Boc and tert-butyl ester) are removed
    in sequence to enable selective transformations.
    """
    
    def __init__(self, config):
        self.strategy_type = config.get("strategy_type", "sequential_deprotection")
        self.group_types = config.get("group_types", ["Boc", "tert-butyl_ester"])
        self.timing = config.get("timing", "staged")
        
        # SMARTS patterns for protecting group detection
        self.boc_pattern = Chem.MolFromSmarts("[NH1][C](=O)OC(C)(C)C")  # Boc carbamate
        self.tert_butyl_ester_pattern = Chem.MolFromSmarts("[C](=O)OC(C)(C)C")  # tert-butyl ester
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        boc_deprotection_depth = -1
        tbu_ester_deprotection_depth = -1
        
        # Find depths of protecting group removals
        for i, rxn in enumerate(reactions):
            if self.detect_boc_deprotection(rxn):
                boc_deprotection_depth = i
            if self.detect_tbu_ester_deprotection(rxn):
                tbu_ester_deprotection_depth = i
        
        # Check if both deprotections occur and in correct sequence
        both_present = (boc_deprotection_depth >= 0 and tbu_ester_deprotection_depth >= 0)
        correct_sequence = boc_deprotection_depth < tbu_ester_deprotection_depth
        
        # For staged timing, require at least one reaction between deprotections
        if self.timing == "staged":
            staged_timing = (tbu_ester_deprotection_depth - boc_deprotection_depth) >= 2
        else:
            staged_timing = True
            
        condition = both_present and correct_sequence and staged_timing
        
        # Return depth as the later deprotection step
        depth = max(boc_deprotection_depth, tbu_ester_deprotection_depth) if both_present else -1
        
        return condition, depth
    
    def detect_boc_deprotection(self, rxn):
        """Detect Boc deprotection by checking if Boc group is lost"""
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[0].split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in rxn_parts[1].split(".")]
            
            if not all(reactants) or not all(products):
                return False
            
            # Count Boc groups in reactants and products
            boc_in_reactants = sum(len(mol.GetSubstructMatches(self.boc_pattern)) for mol in reactants)
            boc_in_products = sum(len(mol.GetSubstructMatches(self.boc_pattern)) for mol in products)
            
            # Boc deprotection occurs if Boc groups decrease
            return boc_in_reactants > boc_in_products
            
        except:
            return False
    
    def detect_tbu_ester_deprotection(self, rxn):
        """Detect tert-butyl ester deprotection by checking if tBu ester group is lost"""
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[0].split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in rxn_parts[1].split(".")]
            
            if not all(reactants) or not all(products):
                return False
            
            # Count tert-butyl ester groups in reactants and products
            tbu_in_reactants = sum(len(mol.GetSubstructMatches(self.tert_butyl_ester_pattern)) for mol in reactants)
            tbu_in_products = sum(len(mol.GetSubstructMatches(self.tert_butyl_ester_pattern)) for mol in products)
            
            # tBu ester deprotection occurs if tBu ester groups decrease
            return tbu_in_reactants > tbu_in_products
            
        except:
            return False
