"""Generated evaluation code for: Multiple protecting group incompatibility issues"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ProtectingGroupIncompatibility(MultiRxnCondBase):
    """
    Detects routes with multiple incompatible protecting groups that would cleave 
    under the same conditions, causing selectivity issues.
    """
    
    def __init__(self, config):
        self.protecting_groups = config["protecting_groups"]
        self.incompatibility_type = config["incompatibility_type"]
        
        # Define SMARTS patterns for protecting groups
        self.pg_patterns = {
            "dimethyl_acetal": "[CH](OC)(OC)",  # Dimethyl acetal
            "trityl": "[CH2]Oc1ccc(cc1)C(c2ccccc2)(c3ccccc3)",  # Trityl ether
            "tert_butyl_ester": "C(=O)OC(C)(C)C",  # tert-Butyl ester
            "boc": "C(=O)OC(C)(C)C",  # tert-Butoxycarbonyl
            "cbz": "C(=O)OCc1ccccc1",  # Benzyloxycarbonyl
            "silyl_ether": "[Si](C)(C)C",  # Trimethylsilyl
            "benzyl_ether": "OCc1ccccc1",  # Benzyl ether
            "pmb": "OCc1ccc(OC)cc1"  # para-Methoxybenzyl
        }
        
        # Define incompatibility groups
        self.incompatible_groups = {
            "acid_labile_clash": ["dimethyl_acetal", "trityl", "tert_butyl_ester", "boc"],
            "base_labile_clash": ["tert_butyl_ester", "cbz"],
            "hydrogenolysis_clash": ["cbz", "benzyl_ether", "pmb"]
        }
    
    def condition_depth(self, d):
        """Check if route contains incompatible protecting groups"""
        reactions = self.get_rxns(d)
        
        # Track which protecting groups are present across all reactions
        present_groups = set()
        
        for rxn in reactions:
            for pg_name in self.protecting_groups:
                if pg_name in self.pg_patterns:
                    if self.detect_protecting_group(rxn, pg_name):
                        present_groups.add(pg_name)
        
        # Check for incompatibility
        incompatible_set = set(self.incompatible_groups.get(self.incompatibility_type, []))
        clash_groups = present_groups.intersection(incompatible_set)
        
        # Condition is met (problematic) if 2+ incompatible groups are present
        condition_met = len(clash_groups) >= 2
        
        return condition_met, len(reactions)
    
    def detect_protecting_group(self, rxn, pg_name):
        """Detect if a specific protecting group is present in the reaction"""
        pattern = self.pg_patterns.get(pg_name)
        if not pattern:
            return False
            
        return self.detect_specific_pattern(rxn, pattern)
    
    def detect_specific_pattern(self, rxn, smarts_pattern):
        """Check if SMARTS pattern appears in any molecule in the reaction"""
        try:
            # Parse reaction SMILES
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants = rxn_parts[0].split(".")
            products = rxn_parts[1].split(".")
            
            pattern_mol = Chem.MolFromSmarts(smarts_pattern)
            if pattern_mol is None:
                return False
            
            # Check all molecules in reaction
            all_mols = reactants + products
            for mol_smiles in all_mols:
                mol = Chem.MolFromSmiles(mol_smiles)
                if mol and mol.HasSubstructMatch(pattern_mol):
                    return True
                    
            return False
            
        except Exception:
            return False
    
    def route_scoring(self, x):
        """Convert condition result to score. Higher score = more problematic"""
        if x < 0:
            return 0  # No incompatibility found
        else:
            return 8  # High penalty for incompatible protecting groups
