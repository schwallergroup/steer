"""Generated evaluation code for: Temporary Boc protection for selective acylation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BocProtectionStrategy(MultiRxnCondBase):
    """
    Evaluates synthesis routes for temporary Boc protection strategy enabling selective acylation.
    Checks for the presence of Boc protection on primary amines followed by selective acylation.
    """
    
    def __init__(self, config):
        self.require_boc_protection = config.get("require_boc_protection", True)
        self.require_selective_acylation = config.get("require_selective_acylation", True)
        self.require_deprotection = config.get("require_deprotection", True)
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        has_boc_protection = any(self.detect_boc_protection(r) for r in reactions)
        has_selective_acylation = any(self.detect_selective_acylation(r) for r in reactions)
        has_boc_deprotection = any(self.detect_boc_deprotection(r) for r in reactions)
        
        # Check if the strategy is executed in logical order
        strategy_complete = self.check_protection_strategy_order(reactions)
        
        condition = True
        if self.require_boc_protection:
            condition = condition and has_boc_protection
        if self.require_selective_acylation:
            condition = condition and has_selective_acylation
        if self.require_deprotection:
            condition = condition and has_boc_deprotection
            
        condition = condition and strategy_complete
        
        return condition, len(reactions)
    
    def detect_boc_protection(self, rxn):
        """Detect Boc protection of primary amine"""
        reactants, products = rxn.split(">>")
        reactant_mols = [Chem.MolFromSmiles(r) for r in reactants.split(".")]
        product_mols = [Chem.MolFromSmiles(p) for p in products.split(".")]
        
        # Primary amine pattern
        primary_amine = Chem.MolFromSmarts("[NH2]")
        # Boc-protected amine pattern
        boc_amine = Chem.MolFromSmarts("[NH1]C(=O)OC(C)(C)C")
        
        # Check if primary amine in reactants becomes Boc-protected in products
        has_primary_amine_reactant = any(mol.HasSubstructMatch(primary_amine) for mol in reactant_mols if mol)
        has_boc_product = any(mol.HasSubstructMatch(boc_amine) for mol in product_mols if mol)
        
        # Also check for Boc reagent in reactants
        boc_reagent = Chem.MolFromSmarts("C(=O)OC(C)(C)C")
        has_boc_reagent = any(mol.HasSubstructMatch(boc_reagent) for mol in reactant_mols if mol)
        
        return has_primary_amine_reactant and has_boc_product and has_boc_reagent
    
    def detect_selective_acylation(self, rxn):
        """Detect selective acylation in presence of Boc-protected amine"""
        reactants, products = rxn.split(">>")
        reactant_mols = [Chem.MolFromSmiles(r) for r in reactants.split(".")]
        product_mols = [Chem.MolFromSmiles(p) for p in products.split(".")]
        
        # Boc-protected amine should be present in reactants
        boc_amine = Chem.MolFromSmarts("[NH1]C(=O)OC(C)(C)C")
        has_boc_reactant = any(mol.HasSubstructMatch(boc_amine) for mol in reactant_mols if mol)
        
        # Look for acylation patterns (amide formation)
        acyl_reagent = Chem.MolFromSmarts("C(=O)[Cl,F,Br,I,OH]")  # Acyl chloride, acid, etc.
        has_acyl_reagent = any(mol.HasSubstructMatch(acyl_reagent) for mol in reactant_mols if mol)
        
        # New amide bond formation
        amide_bond = Chem.MolFromSmarts("NC(=O)")
        reactant_amides = sum(mol.GetSubstructMatches(amide_bond).__len__() for mol in reactant_mols if mol)
        product_amides = sum(mol.GetSubstructMatches(amide_bond).__len__() for mol in product_mols if mol)
        
        new_amide_formed = product_amides > reactant_amides
        
        return has_boc_reactant and has_acyl_reagent and new_amide_formed
    
    def detect_boc_deprotection(self, rxn):
        """Detect Boc deprotection to restore primary amine"""
        reactants, products = rxn.split(">>")
        reactant_mols = [Chem.MolFromSmiles(r) for r in reactants.split(".")]
        product_mols = [Chem.MolFromSmiles(p) for p in products.split(".")]
        
        # Boc-protected amine in reactants
        boc_amine = Chem.MolFromSmarts("[NH1]C(=O)OC(C)(C)C")
        has_boc_reactant = any(mol.HasSubstructMatch(boc_amine) for mol in reactant_mols if mol)
        
        # Primary or secondary amine in products (deprotected)
        free_amine = Chem.MolFromSmarts("[NH2,NH1]")
        has_free_amine_product = any(mol.HasSubstructMatch(free_amine) for mol in product_mols if mol)
        
        # Deprotection reagent (acid)
        acid_reagent = Chem.MolFromSmarts("[H+],HCl,H2SO4,CF3COOH")
        has_acid = any(mol.HasSubstructMatch(acid_reagent) for mol in reactant_mols if mol)
        
        return has_boc_reactant and has_free_amine_product and has_acid
    
    def check_protection_strategy_order(self, reactions):
        """Check if protection, acylation, and deprotection occur in logical sequence"""
        protection_steps = []
        acylation_steps = []
        deprotection_steps = []
        
        for i, rxn in enumerate(reactions):
            if self.detect_boc_protection(rxn):
                protection_steps.append(i)
            if self.detect_selective_acylation(rxn):
                acylation_steps.append(i)
            if self.detect_boc_deprotection(rxn):
                deprotection_steps.append(i)
        
        # Check logical order: protection < acylation < deprotection
        if not (protection_steps and acylation_steps):
            return False
            
        earliest_protection = min(protection_steps) if protection_steps else float('inf')
        earliest_acylation = min(acylation_steps) if acylation_steps else float('inf')
        earliest_deprotection = min(deprotection_steps) if deprotection_steps else float('inf')
        
        # Protection should come before acylation
        if earliest_protection >= earliest_acylation:
            return False
            
        # If deprotection is required, it should come after acylation
        if self.require_deprotection and deprotection_steps:
            if earliest_deprotection <= earliest_acylation:
                return False
                
        return True
