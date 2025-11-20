"""Generated evaluation code for: TES silyl ether protecting group strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class TESProtectingGroupStrategy(MultiRxnCondBase):
    """
    Evaluates whether the synthesis route employs TES (triethylsilyl) protecting group strategy.
    Checks for the presence of TES-protected alcohols and their strategic use throughout the route.
    """
    
    def __init__(self, config):
        self.protecting_group = config["parameters"]["protecting_group"]
        self.functional_group = config["parameters"]["functional_group"] 
        self.tes_smarts = config["parameters"]["smarts"]  # "[Si](CC)(CC)O[C]"
        self.require_strategy = config.get("require_strategy", True)
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Check for TES protection/deprotection reactions
        has_tes_protection = any(self.detect_tes_protection(r) for r in reactions)
        has_tes_deprotection = any(self.detect_tes_deprotection(r) for r in reactions)
        
        # Check for presence of TES-protected intermediates
        has_tes_intermediates = any(self.has_tes_group(r) for r in reactions)
        
        # Strategy requires both protection and deprotection, or persistent intermediates
        if self.require_strategy:
            condition = (has_tes_protection and has_tes_deprotection) or \
                       (has_tes_intermediates and len([r for r in reactions if self.has_tes_group(r)]) >= 2)
        else:
            condition = has_tes_protection or has_tes_deprotection or has_tes_intermediates
            
        return condition, len(reactions)
    
    def detect_tes_protection(self, rxn):
        """Detect TES protection reaction: alcohol + TES reagent -> TES ether"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        # Check if TES group appears in products but not in reactants
        reactant_mols = [Chem.MolFromSmiles(s.strip()) for s in reactants.split(".") if s.strip()]
        product_mols = [Chem.MolFromSmiles(s.strip()) for s in products.split(".") if s.strip()]
        
        if not all(reactant_mols) or not all(product_mols):
            return False
            
        tes_pattern = Chem.MolFromSmarts(self.tes_smarts)
        if not tes_pattern:
            return False
            
        # TES group should be absent in reactants but present in products
        reactants_have_tes = any(mol.HasSubstructMatch(tes_pattern) for mol in reactant_mols)
        products_have_tes = any(mol.HasSubstructMatch(tes_pattern) for mol in product_mols)
        
        return not reactants_have_tes and products_have_tes
    
    def detect_tes_deprotection(self, rxn):
        """Detect TES deprotection reaction: TES ether -> alcohol"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        reactant_mols = [Chem.MolFromSmiles(s.strip()) for s in reactants.split(".") if s.strip()]
        product_mols = [Chem.MolFromSmiles(s.strip()) for s in products.split(".") if s.strip()]
        
        if not all(reactant_mols) or not all(product_mols):
            return False
            
        tes_pattern = Chem.MolFromSmarts(self.tes_smarts)
        if not tes_pattern:
            return False
            
        # TES group should be present in reactants but absent in products
        reactants_have_tes = any(mol.HasSubstructMatch(tes_pattern) for mol in reactant_mols)
        products_have_tes = any(mol.HasSubstructMatch(tes_pattern) for mol in product_mols)
        
        return reactants_have_tes and not products_have_tes
    
    def has_tes_group(self, rxn):
        """Check if any molecule in the reaction contains TES protecting group"""
        rxn_parts = rxn.split(">>")
        all_smiles = []
        
        for part in rxn_parts:
            all_smiles.extend([s.strip() for s in part.split(".") if s.strip()])
            
        tes_pattern = Chem.MolFromSmarts(self.tes_smarts)
        if not tes_pattern:
            return False
            
        for smiles in all_smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol and mol.HasSubstructMatch(tes_pattern):
                return True
                
        return False
