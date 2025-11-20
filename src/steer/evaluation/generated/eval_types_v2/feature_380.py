"""Generated evaluation code for: Sequential protecting group strategy with Cbz"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class CbzSequentialProtection(MultiRxnCondBase):
    """
    Evaluates sequential Cbz protecting group strategy for amines.
    Checks if Cbz protection is applied early in synthesis and removed late,
    providing amine protection throughout multiple synthetic steps.
    """
    
    def __init__(self, config):
        self.protecting_group = config.get("protecting_group", "Cbz")
        self.strategy_type = config.get("strategy_type", "sequential")
        self.functional_group = config.get("functional_group", "amine")
        self.min_protection_span = config.get("min_protection_span", 2)  # Minimum steps between protection/deprotection
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        total_reactions = len(reactions)
        
        if total_reactions < 2:
            return False, total_reactions
            
        protection_step = -1
        deprotection_step = -1
        
        # Find Cbz protection and deprotection steps
        for i, rxn in enumerate(reactions):
            if self.detect_cbz_protection(rxn):
                if protection_step == -1:  # First protection found
                    protection_step = i
            elif self.detect_cbz_deprotection(rxn):
                deprotection_step = i
                
        # Check if sequential strategy is satisfied
        condition = (protection_step != -1 and 
                    deprotection_step != -1 and
                    protection_step < deprotection_step and
                    (deprotection_step - protection_step) >= self.min_protection_span)
        
        return condition, total_reactions
        
    def detect_cbz_protection(self, rxn):
        """Detect Cbz protection of amine groups"""
        reactants_smiles, product_smiles = rxn.split(">>")
        
        try:
            product_mol = Chem.MolFromSmiles(product_smiles)
            reactant_mols = [Chem.MolFromSmiles(r) for r in reactants_smiles.split(".")]
            
            if not product_mol or not all(reactant_mols):
                return False
                
            # Cbz protection pattern: benzyloxycarbonyl attached to nitrogen
            cbz_pattern = Chem.MolFromSmarts("[NH1,NH2]-C(=O)-O-C-c1ccccc1")
            free_amine_pattern = Chem.MolFromSmarts("[NH2,NH3+]")
            
            # Check if product has Cbz-protected amine
            has_cbz_product = product_mol.HasSubstructMatch(cbz_pattern)
            
            # Check if reactants have free amine
            has_free_amine_reactant = any(mol.HasSubstructMatch(free_amine_pattern) 
                                        for mol in reactant_mols)
            
            # Also check for Cbz-Cl reagent in reactants
            cbz_reagent_pattern = Chem.MolFromSmarts("ClC(=O)OCc1ccccc1")
            has_cbz_reagent = any(mol.HasSubstructMatch(cbz_reagent_pattern) 
                                for mol in reactant_mols)
            
            return has_cbz_product and has_free_amine_reactant and has_cbz_reagent
            
        except:
            return False
            
    def detect_cbz_deprotection(self, rxn):
        """Detect Cbz deprotection to reveal amine groups"""
        reactants_smiles, product_smiles = rxn.split(">>")
        
        try:
            product_mol = Chem.MolFromSmiles(product_smiles)
            reactant_mols = [Chem.MolFromSmiles(r) for r in reactants_smiles.split(".")]
            
            if not product_mol or not all(reactant_mols):
                return False
                
            # Cbz protection pattern
            cbz_pattern = Chem.MolFromSmarts("[NH1]-C(=O)-O-C-c1ccccc1")
            free_amine_pattern = Chem.MolFromSmarts("[NH2,NH3+]")
            
            # Check if reactant has Cbz-protected amine
            has_cbz_reactant = any(mol.HasSubstructMatch(cbz_pattern) 
                                 for mol in reactant_mols)
            
            # Check if product has free amine
            has_free_amine_product = product_mol.HasSubstructMatch(free_amine_pattern)
            
            return has_cbz_reactant and has_free_amine_product
            
        except:
            return False
