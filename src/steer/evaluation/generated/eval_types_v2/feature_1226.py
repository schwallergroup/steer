"""Generated evaluation code for: Sequential acid-labile protecting group strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SequentialAcidLabileProtectingGroup(MultiRxnCondBase):
    """
    Evaluates sequential acid-labile protecting group strategy.
    Checks if both Boc and benzhydryl protecting groups are removed 
    sequentially using acid conditions in the synthesis route.
    """
    
    def __init__(self, config):
        self.protecting_groups = config.get("protecting_groups", ["Boc", "benzhydryl"])
        self.deprotection_pattern = config.get("deprotection_pattern", "sequential")
        self.lability_type = config.get("lability_type", "acid")
        
        # Define SMARTS patterns for protecting groups
        self.pg_patterns = {
            "Boc": "[NX3][CX3](=[OX1])[OX2][CX4]([CH3])([CH3])[CH3]",  # Boc group
            "benzhydryl": "[NX3][CHX4]([cX3]1[cX3][cX3][cX3][cX3][cX3]1)[cX3]2[cX3][cX3][cX3][cX3][cX3]2"  # Benzhydryl group
        }
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Track deprotection reactions for each protecting group
        boc_deprotections = []
        benzhydryl_deprotections = []
        
        for i, rxn in enumerate(reactions):
            if self.is_acid_deprotection(rxn):
                if self.detect_boc_deprotection(rxn):
                    boc_deprotections.append(i)
                if self.detect_benzhydryl_deprotection(rxn):
                    benzhydryl_deprotections.append(i)
        
        # Check if both protecting groups are deprotected
        has_both_deprotections = len(boc_deprotections) > 0 and len(benzhydryl_deprotections) > 0
        
        # Check if deprotections are sequential (not simultaneous)
        is_sequential = True
        if has_both_deprotections:
            # Ensure no overlap in reaction indices (sequential, not simultaneous)
            boc_set = set(boc_deprotections)
            benzhydryl_set = set(benzhydryl_deprotections)
            is_sequential = len(boc_set.intersection(benzhydryl_set)) == 0
        
        condition = has_both_deprotections and is_sequential
        return condition, len(reactions)
    
    def detect_boc_deprotection(self, rxn):
        """Detect Boc group removal in reaction"""
        pattern = Chem.MolFromSmarts(self.pg_patterns["Boc"])
        return self.detect_protecting_group_removal(rxn, pattern)
    
    def detect_benzhydryl_deprotection(self, rxn):
        """Detect benzhydryl group removal in reaction"""
        pattern = Chem.MolFromSmarts(self.pg_patterns["benzhydryl"])
        return self.detect_protecting_group_removal(rxn, pattern)
    
    def detect_protecting_group_removal(self, rxn, pattern):
        """Generic method to detect protecting group removal"""
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            products = rxn_parts[0].split(".")
            reactants = rxn_parts[1].split(".")
            
            # Check if protecting group is present in reactants but not in products
            pg_in_reactants = any(
                Chem.MolFromSmiles(r) and Chem.MolFromSmiles(r).HasSubstructMatch(pattern)
                for r in reactants if r.strip()
            )
            
            pg_in_products = any(
                Chem.MolFromSmiles(p) and Chem.MolFromSmiles(p).HasSubstructMatch(pattern)
                for p in products if p.strip()
            )
            
            return pg_in_reactants and not pg_in_products
            
        except:
            return False
    
    def is_acid_deprotection(self, rxn):
        """Check if reaction involves acid conditions"""
        # Common acid reagents and conditions
        acid_patterns = [
            "O=S(=O)(O)O",  # H2SO4
            "Cl",  # HCl
            "F[B-](F)(F)F",  # HBF4
            "O=S(=O)(O)c1ccc(C)cc1",  # TsOH (p-toluenesulfonic acid)
            "[H+]"  # Generic acid
        ]
        
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants = rxn_parts[1].split(".")
            
            for acid_smarts in acid_patterns:
                acid_pattern = Chem.MolFromSmarts(acid_smarts)
                if acid_pattern:
                    for reactant in reactants:
                        mol = Chem.MolFromSmiles(reactant.strip())
                        if mol and mol.HasSubstructMatch(acid_pattern):
                            return True
            
            # Also check for common acid names in reaction string
            acid_names = ["HCl", "H2SO4", "TFA", "HBF4", "TsOH", "acid"]
            rxn_lower = rxn.lower()
            return any(acid_name.lower() in rxn_lower for acid_name in acid_names)
            
        except:
            return False
