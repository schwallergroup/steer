"""Generated evaluation code for: Boc protection strategy for cyclic amidine"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BocCyclicAmidineProtection(MultiRxnCondBase):
    """
    Evaluates synthesis routes for proper Boc protection strategy of cyclic amidines.
    Checks for the presence of both Boc protection and deprotection reactions
    involving cyclic amidine functional groups.
    """
    
    def __init__(self, config):
        self.step_count = config.get("step_count", 2)
        # SMARTS pattern for cyclic amidine (5 or 6-membered rings)
        self.cyclic_amidine_pattern = "[NX2]1[CX3](=[NX2])[CX4][CX4][CX4]1,[NX2]1[CX3](=[NX2])[CX4][CX4][CX4][CX4]1"
        # SMARTS pattern for Boc-protected amidine
        self.boc_protected_pattern = "[NX3]([CX3](=[OX1])[OX2][CX4]([CX4])([CX4])[CX4])[CX3]=[NX2]"
        # SMARTS pattern for Boc group
        self.boc_group_pattern = "[CX3](=[OX1])[OX2][CX4]([CX4])([CX4])[CX4]"
    
    def condition_depth(self, d):
        """
        Check if the route contains both Boc protection and deprotection 
        of cyclic amidine within the expected step count.
        """
        reactions = self.get_rxns(d)
        
        protection_found = any(self.detect_boc_protection(r) for r in reactions)
        deprotection_found = any(self.detect_boc_deprotection(r) for r in reactions)
        
        # Both protection and deprotection should be present
        condition = protection_found and deprotection_found
        
        return condition, len(reactions)
    
    def detect_boc_protection(self, rxn):
        """
        Detect Boc protection reaction: cyclic amidine + Boc reagent -> Boc-protected amidine
        """
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants = rxn_parts[0].split(".")
            products = rxn_parts[1].split(".")
            
            # Check if reactants contain cyclic amidine and Boc reagent
            has_cyclic_amidine = False
            has_boc_reagent = False
            
            for reactant_smiles in reactants:
                mol = Chem.MolFromSmiles(reactant_smiles)
                if mol is None:
                    continue
                    
                # Check for cyclic amidine
                amidine_mol = Chem.MolFromSmarts(self.cyclic_amidine_pattern)
                if amidine_mol and mol.HasSubstructMatch(amidine_mol):
                    has_cyclic_amidine = True
                
                # Check for Boc reagent (Boc2O or similar)
                if "OC(=O)OC(C)(C)C" in reactant_smiles or "Boc" in reactant_smiles:
                    has_boc_reagent = True
            
            # Check if products contain Boc-protected amidine
            has_protected_product = False
            for product_smiles in products:
                mol = Chem.MolFromSmiles(product_smiles)
                if mol is None:
                    continue
                    
                protected_mol = Chem.MolFromSmarts(self.boc_protected_pattern)
                if protected_mol and mol.HasSubstructMatch(protected_mol):
                    has_protected_product = True
                    break
            
            return has_cyclic_amidine and has_boc_reagent and has_protected_product
            
        except Exception:
            return False
    
    def detect_boc_deprotection(self, rxn):
        """
        Detect Boc deprotection reaction: Boc-protected amidine -> cyclic amidine + Boc waste
        """
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants = rxn_parts[0].split(".")
            products = rxn_parts[1].split(".")
            
            # Check if reactants contain Boc-protected amidine
            has_protected_reactant = False
            for reactant_smiles in reactants:
                mol = Chem.MolFromSmiles(reactant_smiles)
                if mol is None:
                    continue
                    
                protected_mol = Chem.MolFromSmarts(self.boc_protected_pattern)
                if protected_mol and mol.HasSubstructMatch(protected_mol):
                    has_protected_reactant = True
                    break
            
            # Check if products contain free cyclic amidine
            has_free_amidine = False
            for product_smiles in products:
                mol = Chem.MolFromSmiles(product_smiles)
                if mol is None:
                    continue
                    
                amidine_mol = Chem.MolFromSmarts(self.cyclic_amidine_pattern)
                if amidine_mol and mol.HasSubstructMatch(amidine_mol):
                    # Ensure it's not still Boc-protected
                    protected_mol = Chem.MolFromSmarts(self.boc_protected_pattern)
                    if not (protected_mol and mol.HasSubstructMatch(protected_mol)):
                        has_free_amidine = True
                        break
            
            return has_protected_reactant and has_free_amidine
            
        except Exception:
            return False
