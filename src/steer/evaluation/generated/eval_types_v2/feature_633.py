"""Generated evaluation code for: Protection-deprotection cycling for alcohol"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class AlcoholProtectionCycling(MultiRxnCondBase):
    """
    Detects protection-deprotection cycling for alcohol functional groups.
    Specifically looks for acetate protection of primary alcohols followed by deprotection.
    """
    
    def __init__(self, config):
        self.functional_group = config.get("functional_group", "alcohol")
        self.protecting_group = config.get("protecting_group", "acetate")
        self.sequence_type = config.get("sequence_type", "cycling")
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Track protection and deprotection events
        protection_found = False
        deprotection_found = False
        protection_depth = -1
        deprotection_depth = -1
        
        for i, rxn in enumerate(reactions):
            if self.detect_acetate_protection(rxn):
                protection_found = True
                protection_depth = i
            elif self.detect_acetate_deprotection(rxn):
                deprotection_found = True
                deprotection_depth = i
        
        # For cycling, we need both protection and deprotection
        # and deprotection should occur after protection
        if self.sequence_type == "cycling":
            condition = (protection_found and deprotection_found and 
                        protection_depth < deprotection_depth)
        else:
            condition = protection_found and deprotection_found
        
        # Return the depth of the later event (deprotection for cycling)
        depth = max(protection_depth, deprotection_depth) if condition else -1
        
        return condition, depth
    
    def detect_acetate_protection(self, rxn):
        """Detect protection of primary alcohol with acetate"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0].split(".")
        products = rxn_parts[1].split(".")
        
        # Primary alcohol pattern
        primary_alcohol_pattern = "[CH2][OH]"
        # Acetate ester pattern
        acetate_pattern = "[CH2]OC(=O)C"
        
        # Check if we go from primary alcohol to acetate ester
        has_alcohol_reactant = any(
            self.has_substructure(r, primary_alcohol_pattern) for r in reactants
        )
        has_acetate_product = any(
            self.has_substructure(p, acetate_pattern) for p in products
        )
        
        return has_alcohol_reactant and has_acetate_product
    
    def detect_acetate_deprotection(self, rxn):
        """Detect deprotection of acetate to regenerate alcohol"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0].split(".")
        products = rxn_parts[1].split(".")
        
        # Acetate ester pattern
        acetate_pattern = "[CH2]OC(=O)C"
        # Primary alcohol pattern
        primary_alcohol_pattern = "[CH2][OH]"
        
        # Check if we go from acetate ester back to primary alcohol
        has_acetate_reactant = any(
            self.has_substructure(r, acetate_pattern) for r in reactants
        )
        has_alcohol_product = any(
            self.has_substructure(p, primary_alcohol_pattern) for p in products
        )
        
        return has_acetate_reactant and has_alcohol_product
    
    def has_substructure(self, smiles, pattern):
        """Check if molecule contains substructure pattern"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            pattern_mol = Chem.MolFromSmarts(pattern)
            if mol is None or pattern_mol is None:
                return False
            return mol.HasSubstructMatch(pattern_mol)
        except:
            return False
