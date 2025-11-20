"""Generated evaluation code for: Azide reduction before Boc protection"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class AzideReductionBocProtection(MultiRxnCondBase):
    """
    Checks for the presence of azide reduction followed by Boc protection in the synthesis route.
    Detects the specific sequence where an azide group is reduced to an amine and then 
    protected with a Boc group.
    """
    
    def __init__(self, config):
        self.require_sequence = config.get("require_sequence", True)
        self.allow_intervening_steps = config.get("allow_intervening_steps", 1)
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        azide_reduction_found = False
        boc_protection_found = False
        sequence_correct = False
        
        # Check for individual reaction types
        azide_indices = []
        boc_indices = []
        
        for i, rxn in enumerate(reactions):
            if self.detect_azide_reduction(rxn):
                azide_reduction_found = True
                azide_indices.append(i)
            if self.detect_boc_protection(rxn):
                boc_protection_found = True
                boc_indices.append(i)
        
        # Check sequence if both reactions are present
        if azide_reduction_found and boc_protection_found and self.require_sequence:
            sequence_correct = self.check_sequence(azide_indices, boc_indices)
        elif azide_reduction_found and boc_protection_found:
            sequence_correct = True
        
        condition = azide_reduction_found and boc_protection_found and sequence_correct
        return condition, len(reactions)
    
    def detect_azide_reduction(self, rxn):
        """Detect azide reduction to amine"""
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
            
            reactants = rxn_parts[0]
            products = rxn_parts[1]
            
            # Check for azide in reactants and amine in products
            azide_pattern = Chem.MolFromSmarts("[N-]=[N+]=[N-]")  # Azide group
            amine_pattern = Chem.MolFromSmarts("[NH2,NH1]")  # Primary or secondary amine
            
            reactant_mols = [Chem.MolFromSmiles(mol.strip()) for mol in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(mol.strip()) for mol in products.split(".")]
            
            # Filter out None molecules
            reactant_mols = [mol for mol in reactant_mols if mol is not None]
            product_mols = [mol for mol in product_mols if mol is not None]
            
            # Check for azide in reactants
            azide_in_reactants = any(mol.HasSubstructMatch(azide_pattern) for mol in reactant_mols)
            
            # Check for amine in products
            amine_in_products = any(mol.HasSubstructMatch(amine_pattern) for mol in product_mols)
            
            return azide_in_reactants and amine_in_products
            
        except Exception:
            return False
    
    def detect_boc_protection(self, rxn):
        """Detect Boc protection of amine"""
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
            
            reactants = rxn_parts[0]
            products = rxn_parts[1]
            
            # Check for free amine in reactants and Boc-protected amine in products
            amine_pattern = Chem.MolFromSmarts("[NH2,NH1]")  # Primary or secondary amine
            boc_pattern = Chem.MolFromSmarts("CC(C)(C)OC(=O)N")  # Boc protecting group
            
            reactant_mols = [Chem.MolFromSmiles(mol.strip()) for mol in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(mol.strip()) for mol in products.split(".")]
            
            # Filter out None molecules
            reactant_mols = [mol for mol in reactant_mols if mol is not None]
            product_mols = [mol for mol in product_mols if mol is not None]
            
            # Check for free amine in reactants
            amine_in_reactants = any(mol.HasSubstructMatch(amine_pattern) for mol in reactant_mols)
            
            # Check for Boc group in products
            boc_in_products = any(mol.HasSubstructMatch(boc_pattern) for mol in product_mols)
            
            return amine_in_reactants and boc_in_products
            
        except Exception:
            return False
    
    def check_sequence(self, azide_indices, boc_indices):
        """Check if Boc protection occurs after azide reduction within allowed steps"""
        for azide_idx in azide_indices:
            for boc_idx in boc_indices:
                if 0 <= (boc_idx - azide_idx) <= self.allow_intervening_steps:
                    return True
        return False
