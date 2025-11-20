"""Generated evaluation code for: Dual protecting group strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class DualProtectingGroupStrategy(MultiRxnCondBase):
    """
    Evaluates routes based on dual protecting group strategy usage.
    Checks for simultaneous use of specified protecting groups (SEM and acetal).
    """
    
    def __init__(self, config):
        self.protection_types = config.get("protection_types", ["SEM", "acetal"])
        self.simultaneous_use = config.get("simultaneous_use", True)
        
        # Define SMARTS patterns for protecting groups
        self.protecting_group_patterns = {
            "SEM": "[CH2][O][CH2][Si]([CH3])([CH3])[CH2][CH3]",  # SEM protecting group
            "acetal": "[CH]([O][CH2,CH3])([O][CH2,CH3])",  # Acetal pattern
            "ketal": "[C]([O][CH2,CH3])([O][CH2,CH3])",  # Ketal pattern (alternative acetal)
            "TMS": "[Si]([CH3])([CH3])[CH3]",  # TMS protecting group
            "TBDMS": "[Si]([CH3])([CH3])[C]([CH3])([CH3])[CH3]",  # TBDMS protecting group
            "benzyl": "[CH2]c1ccccc1",  # Benzyl protecting group
            "BOC": "[C]([O][C]([CH3])([CH3])[CH3])(=[O])[N]"  # BOC protecting group
        }

    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Track which protecting groups are detected
        detected_groups = set()
        
        for rxn in reactions:
            for group_type in self.protection_types:
                if self.detect_protecting_group(rxn, group_type):
                    detected_groups.add(group_type)
        
        if self.simultaneous_use:
            # Check if ALL required protecting groups are present
            condition = all(group in detected_groups for group in self.protection_types)
        else:
            # Check if ANY of the required protecting groups are present
            condition = any(group in detected_groups for group in self.protection_types)
        
        return condition, len(reactions)

    def detect_protecting_group(self, rxn, group_type):
        """
        Detect if a specific protecting group is involved in the reaction.
        Looks for protection (formation) or deprotection (removal) steps.
        """
        if group_type not in self.protecting_group_patterns:
            return False
            
        pattern = self.protecting_group_patterns[group_type]
        mol_pattern = Chem.MolFromSmarts(pattern)
        
        if mol_pattern is None:
            return False
        
        # Parse reaction SMILES
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants_smiles = rxn_parts[0]
        products_smiles = rxn_parts[1]
        
        # Check reactants and products for protecting group patterns
        reactant_mols = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
        product_mols = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
        
        # Remove None molecules (parsing failures)
        reactant_mols = [mol for mol in reactant_mols if mol is not None]
        product_mols = [mol for mol in product_mols if mol is not None]
        
        # Count protecting group occurrences in reactants and products
        reactant_matches = sum(len(mol.GetSubstructMatches(mol_pattern)) for mol in reactant_mols)
        product_matches = sum(len(mol.GetSubstructMatches(mol_pattern)) for mol in product_mols)
        
        # Protection: protecting group appears in products but not (as much) in reactants
        # Deprotection: protecting group disappears from reactants to products
        # Either case indicates use of this protecting group strategy
        return reactant_matches != product_matches

    def route_scoring(self, x):
        """
        Convert condition result to score.
        Higher score for meeting the dual protecting group strategy condition.
        """
        if x < 0:
            return 0  # Condition not met
        else:
            return 10  # Condition met - full score
