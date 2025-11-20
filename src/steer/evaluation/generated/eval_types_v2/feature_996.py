"""Generated evaluation code for: Protection deprotection cycling on triazole nitrogen"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class TriazoleBenzylProtectionCycling(MultiRxnCondBase):
    """
    Evaluates protection-deprotection cycling on triazole nitrogen using benzyl groups.
    Checks for the presence of both benzyl protection and deprotection reactions
    involving triazole nitrogen atoms in the synthesis route.
    """
    
    def __init__(self, config):
        self.require_protection = config.get("require_protection", True)
        self.require_deprotection = config.get("require_deprotection", True)
        self.allow_cycling = config.get("allow_cycling", True)
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        has_protection = any(self.detect_benzyl_protection(r) for r in reactions)
        has_deprotection = any(self.detect_benzyl_deprotection(r) for r in reactions)
        
        # Check if both protection and deprotection occur (cycling)
        cycling_occurs = has_protection and has_deprotection
        
        # Evaluate condition based on configuration
        if self.allow_cycling:
            condition = cycling_occurs
        else:
            condition = not cycling_occurs
            
        return condition, len(reactions)
    
    def detect_benzyl_protection(self, rxn):
        """Detect benzyl protection of triazole nitrogen"""
        try:
            reactants_smiles, products_smiles = rxn.split(">>")
            reactants = [Chem.MolFromSmiles(r) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p) for p in products_smiles.split(".")]
            
            # Triazole patterns
            triazole_patterns = [
                Chem.MolFromSmarts("[nH]1ncc-1"),  # 1H-1,2,3-triazole
                Chem.MolFromSmarts("[nH]1ccn-1"),  # 1H-1,2,4-triazole
                Chem.MolFromSmarts("[nH]1cnc-1"),  # other triazole isomer
            ]
            
            # N-benzyl triazole patterns
            benzyl_triazole_patterns = [
                Chem.MolFromSmarts("n1(Cc2ccccc2)ncc-1"),  # N-benzyl-1,2,3-triazole
                Chem.MolFromSmarts("n1(Cc2ccccc2)ccn-1"),  # N-benzyl-1,2,4-triazole
                Chem.MolFromSmarts("n1(Cc2ccccc2)cnc-1"),  # other N-benzyl triazole
            ]
            
            # Check if reactants have free triazole NH
            has_free_triazole = any(
                any(mol.HasSubstructMatch(pattern) for pattern in triazole_patterns)
                for mol in reactants if mol is not None
            )
            
            # Check if products have N-benzyl triazole
            has_benzyl_triazole = any(
                any(mol.HasSubstructMatch(pattern) for pattern in benzyl_triazole_patterns)
                for mol in products if mol is not None
            )
            
            # Also check for benzyl bromide or benzyl chloride reagents
            has_benzyl_reagent = any(
                mol.HasSubstructMatch(Chem.MolFromSmarts("BrCc1ccccc1")) or
                mol.HasSubstructMatch(Chem.MolFromSmarts("ClCc1ccccc1"))
                for mol in reactants if mol is not None
            )
            
            return has_free_triazole and has_benzyl_triazole and has_benzyl_reagent
            
        except:
            return False
    
    def detect_benzyl_deprotection(self, rxn):
        """Detect benzyl deprotection of triazole nitrogen"""
        try:
            reactants_smiles, products_smiles = rxn.split(">>")
            reactants = [Chem.MolFromSmiles(r) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p) for p in products_smiles.split(".")]
            
            # N-benzyl triazole patterns
            benzyl_triazole_patterns = [
                Chem.MolFromSmarts("n1(Cc2ccccc2)ncc-1"),
                Chem.MolFromSmarts("n1(Cc2ccccc2)ccn-1"),
                Chem.MolFromSmarts("n1(Cc2ccccc2)cnc-1"),
            ]
            
            # Free triazole patterns
            triazole_patterns = [
                Chem.MolFromSmarts("[nH]1ncc-1"),
                Chem.MolFromSmarts("[nH]1ccn-1"),
                Chem.MolFromSmarts("[nH]1cnc-1"),
            ]
            
            # Check if reactants have N-benzyl triazole
            has_benzyl_triazole = any(
                any(mol.HasSubstructMatch(pattern) for pattern in benzyl_triazole_patterns)
                for mol in reactants if mol is not None
            )
            
            # Check if products have free triazole NH
            has_free_triazole = any(
                any(mol.HasSubstructMatch(pattern) for pattern in triazole_patterns)
                for mol in products if mol is not None
            )
            
            # Check for typical deprotection conditions (H2/Pd, Na/NH3, etc.)
            has_deprotection_conditions = any(
                mol.HasSubstructMatch(Chem.MolFromSmarts("[Pd]")) or  # Palladium catalyst
                mol.HasSubstructMatch(Chem.MolFromSmarts("[H][H]"))    # Hydrogen gas
                for mol in reactants if mol is not None
            )
            
            return has_benzyl_triazole and has_free_triazole
            
        except:
            return False
