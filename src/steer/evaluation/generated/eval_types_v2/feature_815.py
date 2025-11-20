"""Generated evaluation code for: Boc protecting group strategy for aniline"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BocAnilineProtection(MultiRxnCondBase):
    """
    Evaluates synthesis routes for Boc protecting group strategy on aniline.
    Checks if aniline is protected with Boc, remains protected for specified steps,
    and is then deprotected.
    """
    
    def __init__(self, config):
        self.protecting_group = config.get("protecting_group", "Boc")
        self.functional_group = config.get("functional_group", "aniline")
        self.steps_protected = config.get("steps_protected", 2)
        
        # SMARTS patterns
        self.aniline_pattern = "[NH2]-[c]"  # Aniline: NH2 attached to aromatic carbon
        self.boc_aniline_pattern = "[NH1]([C](=[O])[O][C]([CH3])([CH3])[CH3])-[c]"  # Boc-protected aniline
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        """
        Check if the route contains proper Boc protection strategy:
        1. Aniline gets Boc protection
        2. Remains protected for specified number of steps
        3. Gets deprotected back to aniline
        """
        reactions = self.get_rxns(d)
        
        if len(reactions) < 3:  # Need at least protection + reaction + deprotection
            return False, len(reactions)
        
        protection_found = False
        deprotection_found = False
        protection_step = -1
        deprotection_step = -1
        
        # Look for Boc protection step
        for i, rxn in enumerate(reactions):
            if self.is_boc_protection(rxn):
                protection_found = True
                protection_step = i
                break
        
        if not protection_found:
            return False, len(reactions)
        
        # Look for Boc deprotection step after protection
        for i, rxn in enumerate(reactions[protection_step + 1:], start=protection_step + 1):
            if self.is_boc_deprotection(rxn):
                deprotection_found = True
                deprotection_step = i
                break
        
        if not deprotection_found:
            return False, len(reactions)
        
        # Check if protection lasted for the required number of steps
        protection_duration = deprotection_step - protection_step - 1
        steps_match = protection_duration >= self.steps_protected
        
        return steps_match, len(reactions)
    
    def is_boc_protection(self, rxn):
        """Check if reaction is Boc protection of aniline."""
        reactants_smiles, products_smiles = rxn.split(">>")
        
        # Parse reactants and products
        reactant_mols = [Chem.MolFromSmiles(smi) for smi in reactants_smiles.split(".")]
        product_mols = [Chem.MolFromSmiles(smi) for smi in products_smiles.split(".")]
        
        # Check if any reactant has aniline
        has_aniline_reactant = any(
            mol and mol.HasSubstructMatch(Chem.MolFromSmarts(self.aniline_pattern))
            for mol in reactant_mols
        )
        
        # Check if any product has Boc-protected aniline
        has_boc_product = any(
            mol and mol.HasSubstructMatch(Chem.MolFromSmarts(self.boc_aniline_pattern))
            for mol in product_mols
        )
        
        # Check for Boc reagent in reactants (tert-butoxycarbonyl source)
        boc_reagent_pattern = "[C]([CH3])([CH3])[CH3][O][C](=[O])"  # Boc reagent pattern
        has_boc_reagent = any(
            mol and mol.HasSubstructMatch(Chem.MolFromSmarts(boc_reagent_pattern))
            for mol in reactant_mols
        )
        
        return has_aniline_reactant and has_boc_product and has_boc_reagent
    
    def is_boc_deprotection(self, rxn):
        """Check if reaction is Boc deprotection to give aniline."""
        reactants_smiles, products_smiles = rxn.split(">>")
        
        # Parse reactants and products
        reactant_mols = [Chem.MolFromSmiles(smi) for smi in reactants_smiles.split(".")]
        product_mols = [Chem.MolFromSmiles(smi) for smi in products_smiles.split(".")]
        
        # Check if any reactant has Boc-protected aniline
        has_boc_reactant = any(
            mol and mol.HasSubstructMatch(Chem.MolFromSmarts(self.boc_aniline_pattern))
            for mol in reactant_mols
        )
        
        # Check if any product has free aniline
        has_aniline_product = any(
            mol and mol.HasSubstructMatch(Chem.MolFromSmarts(self.aniline_pattern))
            for mol in product_mols
        )
        
        return has_boc_reactant and has_aniline_product
