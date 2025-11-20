"""Generated evaluation code for: Benzyl to silyl protecting group switching"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BenzylToSilylSwitch(MultiRxnCondBase):
    """
    Evaluates synthesis routes for benzyl to silyl protecting group switching.
    
    Checks if the route contains both:
    1. Benzyl ether deprotection (benzyl group removal from alcohol)
    2. Silyl ether protection (TBDMS/TMS/TIPS protection of alcohol)
    
    Returns higher scores when both operations occur in the same route.
    """
    
    def __init__(self, config):
        self.initial_group = config["parameters"].get("initial_group", "benzyl")
        self.final_group = config["parameters"].get("final_group", "silyl")
        self.functional_group = config["parameters"].get("functional_group", "alcohol")
        self.involves_switching = config["parameters"].get("involves_switching", True)
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        has_benzyl_deprotection = False
        has_silyl_protection = False
        
        for rxn in reactions:
            if self.detect_benzyl_deprotection(rxn):
                has_benzyl_deprotection = True
            if self.detect_silyl_protection(rxn):
                has_silyl_protection = True
        
        # Both conditions must be met for successful switching
        condition = has_benzyl_deprotection and has_silyl_protection
        return condition, len(reactions)
    
    def detect_benzyl_deprotection(self, rxn):
        """Detect benzyl ether deprotection (Bn-O-R -> HO-R)"""
        # Benzyl ether pattern: aromatic ring connected to CH2-O-R
        benzyl_ether_pattern = "c1ccccc1CO[#6]"
        
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        # Check if benzyl ether is present in reactants
        reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".") if r.strip()]
        has_benzyl_reactant = any(
            mol and mol.HasSubstructMatch(Chem.MolFromSmarts(benzyl_ether_pattern)) 
            for mol in reactant_mols
        )
        
        if not has_benzyl_reactant:
            return False
        
        # Check if free alcohol appears in products (and benzyl group is removed)
        product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".") if p.strip()]
        has_free_alcohol = any(
            mol and mol.HasSubstructMatch(Chem.MolFromSmarts("[#6][OH]"))
            for mol in product_mols
        )
        
        # Benzyl group should be absent or separate in products
        benzyl_attached_in_products = any(
            mol and mol.HasSubstructMatch(Chem.MolFromSmarts(benzyl_ether_pattern))
            for mol in product_mols
        )
        
        return has_free_alcohol and not benzyl_attached_in_products
    
    def detect_silyl_protection(self, rxn):
        """Detect silyl ether formation (HO-R -> silyl-O-R)"""
        # Common silyl protecting groups: TBDMS, TMS, TIPS
        silyl_patterns = [
            "[Si]([CH3])([CH3])[CH3]O[#6]",  # TMS ether
            "[Si]([CH3])([CH3])([CH3])O[#6]",  # Additional TMS pattern
            "[Si](C(C)(C)C)([CH3])([CH3])O[#6]",  # TBDMS ether
            "[Si]([CH](C)C)([CH](C)C)([CH](C)C)O[#6]"  # TIPS ether
        ]
        
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        # Check if free alcohol is present in reactants
        reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".") if r.strip()]
        has_free_alcohol_reactant = any(
            mol and mol.HasSubstructMatch(Chem.MolFromSmarts("[#6][OH]"))
            for mol in reactant_mols
        )
        
        if not has_free_alcohol_reactant:
            return False
        
        # Check if silyl ether appears in products
        product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".") if p.strip()]
        has_silyl_product = False
        
        for pattern in silyl_patterns:
            if any(mol and mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)) 
                   for mol in product_mols):
                has_silyl_product = True
                break
        
        return has_silyl_product
