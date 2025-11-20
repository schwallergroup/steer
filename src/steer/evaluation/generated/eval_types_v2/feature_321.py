"""Generated evaluation code for: Orthogonal protecting group strategy for phenols"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class OrthogonalProtectingGroupStrategy(MultiRxnCondBase):
    """
    Evaluates synthesis routes for orthogonal protecting group strategies for phenols.
    Checks if multiple different protecting groups (TIPS, MOM, acetate) are used
    for phenolic hydroxyl groups with orthogonal deprotection conditions.
    """
    
    def __init__(self, config):
        self.functional_group = config.get("functional_group", "phenol")
        self.protecting_groups = config.get("protecting_groups", ["TIPS", "MOM", "acetate"])
        self.orthogonal = config.get("orthogonal", True)
        self.min_groups_required = 2 if self.orthogonal else 1
        
        # Define SMARTS patterns for protecting groups
        self.protection_patterns = {
            "TIPS": "[OH1][Si]([CH3])([CH3])[CH]([CH3])([CH3])",  # TIPS-protected phenol
            "MOM": "[OH1][CH2][O][CH3]",  # MOM-protected phenol  
            "acetate": "[OH1][C](=O)[CH3]"  # Acetate-protected phenol
        }
        
        # Phenol pattern for detecting phenolic OH groups
        self.phenol_pattern = "c[OH1]"
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Track which protecting groups are used
        groups_used = set()
        phenol_protections = 0
        
        for rxn in reactions:
            # Check for protection reactions
            protection_type = self.detect_phenol_protection(rxn)
            if protection_type:
                groups_used.add(protection_type)
                phenol_protections += 1
        
        # Evaluate conditions
        if self.orthogonal:
            # Need at least 2 different protecting groups used
            condition = len(groups_used) >= self.min_groups_required and phenol_protections >= 2
        else:
            # Just need at least one of the specified protecting groups
            condition = len(groups_used) >= 1
            
        return condition, len(reactions)
    
    def detect_phenol_protection(self, rxn):
        """
        Detect if a reaction involves protection of a phenolic OH group
        Returns the type of protecting group used, or None
        """
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return None
                
            reactants = rxn_parts[0]
            products = rxn_parts[1]
            
            # Parse molecules
            reactant_mols = [Chem.MolFromSmiles(smi.strip()) for smi in reactants.split('.')]
            product_mols = [Chem.MolFromSmiles(smi.strip()) for smi in products.split('.')]
            
            if not all(reactant_mols) or not all(product_mols):
                return None
            
            # Check if reactants have free phenolic OH
            has_free_phenol_reactant = any(
                mol.HasSubstructMatch(Chem.MolFromSmarts(self.phenol_pattern)) 
                for mol in reactant_mols if mol
            )
            
            if not has_free_phenol_reactant:
                return None
            
            # Check if products have protected phenolic groups
            for group_name, pattern in self.protection_patterns.items():
                if group_name in self.protecting_groups:
                    pattern_mol = Chem.MolFromSmarts(pattern)
                    if pattern_mol and any(
                        mol.HasSubstructMatch(pattern_mol) 
                        for mol in product_mols if mol
                    ):
                        return group_name
            
            return None
            
        except Exception:
            return None
    
    def route_scoring(self, x):
        """
        Convert condition result to 0-10 score.
        Higher score for meeting the orthogonal protection strategy.
        """
        if x < 0:
            return 0  # Strategy not implemented
        else:
            return 10  # Strategy successfully implemented
