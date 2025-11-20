"""Generated evaluation code for: Early stage acetate protection of diol"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class AcetateProtectionStrategy(BaseScoring):
    """
    Evaluates acetate protection strategy for diols, checking if acetate protection
    occurs early in the synthesis and deprotection occurs later.
    """
    
    def __init__(self, config: Dict):
        self.target_protection_depth = config.get("target_protection_depth", 0.8)  # Early stage
        self.require_deprotection = config.get("require_deprotection", True)
        self.protection_depth = -1
        self.deprotection_depth = -1
    
    def route_scoring(self, x) -> float:
        # x is the protection depth fraction
        if x < 0:
            return 0  # No acetate protection found
        
        # Check if we also found deprotection when required
        if self.require_deprotection and self.deprotection_depth < 0:
            return 0  # Protection without deprotection
        
        # Score based on how early the protection occurs
        # Early protection (high depth fraction) gets higher score
        if x >= self.target_protection_depth:
            base_score = 1.0
        else:
            base_score = x / self.target_protection_depth
        
        # Bonus if deprotection occurs later (lower depth)
        if self.deprotection_depth >= 0:
            if self.deprotection_depth < x:  # Deprotection after protection
                base_score *= 1.2
        
        return min(base_score, 1.0)
    
    def hit_condition(self, d):
        """Check if this reaction involves acetate protection of a diol"""
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            rxn_parts = rxn_smiles.split(">>")
            reactants = rxn_parts[0]
            products = rxn_parts[1]
            
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".")]
            
            # Check for acetate protection (diol + acetyl chloride/anhydride -> acetate ester)
            if self._is_acetate_protection(reactant_mols, product_mols):
                return True
            
            # Also track deprotection reactions
            if self._is_acetate_deprotection(reactant_mols, product_mols):
                current_depth = d.get("depth", 0)
                total_depth = d.get("total_depth", 1)
                self.deprotection_depth = current_depth / total_depth
            
            return False
            
        except Exception:
            return False
    
    def _is_acetate_protection(self, reactants, products):
        """Check if reaction is acetate protection of diol"""
        # Diol pattern - two OH groups
        diol_pattern = Chem.MolFromSmarts("[OH][C,c][C,c][OH]")
        
        # Acetylating agent patterns
        acetyl_chloride = Chem.MolFromSmarts("CC(=O)Cl")
        acetic_anhydride = Chem.MolFromSmarts("CC(=O)OC(=O)C")
        
        # Acetate ester pattern
        acetate_pattern = Chem.MolFromSmarts("CC(=O)O[C,c]")
        
        # Check if reactants contain diol and acetylating agent
        has_diol = any(mol and mol.HasSubstructMatch(diol_pattern) for mol in reactants)
        has_acetylating_agent = any(
            mol and (mol.HasSubstructMatch(acetyl_chloride) or mol.HasSubstructMatch(acetic_anhydride))
            for mol in reactants
        )
        
        # Check if products contain acetate ester
        has_acetate_product = any(mol and mol.HasSubstructMatch(acetate_pattern) for mol in products)
        
        return has_diol and has_acetylating_agent and has_acetate_product
    
    def _is_acetate_deprotection(self, reactants, products):
        """Check if reaction is acetate deprotection"""
        # Acetate ester pattern
        acetate_pattern = Chem.MolFromSmarts("CC(=O)O[C,c]")
        
        # Free hydroxyl pattern
        hydroxyl_pattern = Chem.MolFromSmarts("[OH][C,c]")
        
        # Check if reactants have acetate and products have free OH
        has_acetate_reactant = any(mol and mol.HasSubstructMatch(acetate_pattern) for mol in reactants)
        has_hydroxyl_product = any(mol and mol.HasSubstructMatch(hydroxyl_pattern) for mol in products)
        
        return has_acetate_reactant and has_hydroxyl_product
