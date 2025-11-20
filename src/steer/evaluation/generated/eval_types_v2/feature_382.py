"""Generated evaluation code for: Cbz protecting group cycling strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class CbzProtectingGroupStrategy(MultiRxnCondBase):
    """
    Evaluates whether a Cbz protecting group cycling strategy is used effectively.
    Checks if Cbz protection occurs early in the synthesis and deprotection occurs late stage.
    """
    
    def __init__(self, config):
        self.protecting_group = config.get("protecting_group", "Cbz")
        self.operations = config.get("operations", ["protection", "deprotection"])
        self.functional_group = config.get("functional_group", "amine")
        self.early_threshold = config.get("early_threshold", 0.7)  # Protection should occur in first 70%
        self.late_threshold = config.get("late_threshold", 0.3)   # Deprotection should occur in last 30%
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        total_reactions = len(reactions)
        
        if total_reactions == 0:
            return False, 0
        
        protection_depths = []
        deprotection_depths = []
        
        for i, rxn in enumerate(reactions):
            if self.detect_cbz_protection(rxn):
                protection_depths.append(i / total_reactions)
            elif self.detect_cbz_deprotection(rxn):
                deprotection_depths.append(i / total_reactions)
        
        # Check if strategy is properly implemented
        has_early_protection = any(depth >= self.early_threshold for depth in protection_depths)
        has_late_deprotection = any(depth <= self.late_threshold for depth in deprotection_depths)
        
        # Both protection and deprotection should be present for proper cycling
        condition = (len(protection_depths) > 0 and len(deprotection_depths) > 0 and 
                    has_early_protection and has_late_deprotection)
        
        return condition, total_reactions
    
    def detect_cbz_protection(self, rxn):
        """Detect Cbz protection of amine groups"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        try:
            # Cbz reagent pattern (benzyl chloroformate or similar)
            cbz_reagent_patterns = [
                "ClC(=O)OCc1ccccc1",  # Benzyl chloroformate
                "O=C(OCc1ccccc1)OCc1ccccc1",  # Cbz anhydride
            ]
            
            # Check if Cbz reagent is present in reactants
            has_cbz_reagent = False
            for pattern in cbz_reagent_patterns:
                if pattern in reactants or self._contains_substructure(reactants, pattern):
                    has_cbz_reagent = True
                    break
            
            # Check for formation of Cbz-protected amine (carbamate)
            cbz_protected_pattern = "O=C(NCc1ccccc1)OCc1ccccc1"  # Simplified Cbz-protected amine
            has_cbz_product = self._contains_substructure(products, cbz_protected_pattern)
            
            return has_cbz_reagent and has_cbz_product
            
        except:
            return False
    
    def detect_cbz_deprotection(self, rxn):
        """Detect Cbz deprotection reactions"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        try:
            # Cbz-protected amine pattern in reactants
            cbz_protected_pattern = "O=C(N)OCc1ccccc1"  # Cbz carbamate
            has_cbz_reactant = self._contains_substructure(reactants, cbz_protected_pattern)
            
            # Common deprotection conditions
            deprotection_reagents = [
                "[H][H]",  # H2 for hydrogenolysis
                "[Pd]",    # Pd catalyst
                "Br",      # HBr
                "CCO"      # Often in ethanol
            ]
            
            has_deprotection_conditions = any(
                reagent in reactants for reagent in deprotection_reagents
            )
            
            # Check for free amine formation (loss of Cbz group)
            has_free_amine = "N" in products and not self._contains_substructure(products, cbz_protected_pattern)
            
            return has_cbz_reactant and (has_deprotection_conditions or has_free_amine)
            
        except:
            return False
    
    def _contains_substructure(self, smiles_string, pattern):
        """Helper method to check if a pattern exists in SMILES"""
        try:
            from rdkit import Chem
            mols = smiles_string.split(".")
            pattern_mol = Chem.MolFromSmarts(pattern)
            if pattern_mol is None:
                pattern_mol = Chem.MolFromSmiles(pattern)
            
            if pattern_mol is None:
                return False
                
            for mol_smiles in mols:
                mol = Chem.MolFromSmiles(mol_smiles.strip())
                if mol is not None and mol.HasSubstructMatch(pattern_mol):
                    return True
            return False
        except:
            return False
    
    def route_scoring(self, x):
        """Score the route based on proper Cbz cycling strategy implementation"""
        if x < 0:
            return 0  # Strategy not implemented
        else:
            return 10  # Strategy properly implemented
