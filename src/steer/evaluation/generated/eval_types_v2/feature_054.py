"""Generated evaluation code for: Ketal protection without planned deprotection step"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class KetalProtectionWithoutDeprotection(MultiRxnCondBase):
    """
    Evaluates synthesis routes that introduce ketal protection but fail to include
    the necessary deprotection step. Returns a penalty score if ketal protection
    is used without subsequent deprotection.
    """
    
    def __init__(self, config):
        self.deprotection_required = not config["parameters"].get("deprotection_included", True)
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        has_ketal_protection = False
        has_ketal_deprotection = False
        
        for rxn in reactions:
            if self.detect_ketal_protection(rxn):
                has_ketal_protection = True
            if self.detect_ketal_deprotection(rxn):
                has_ketal_deprotection = True
        
        # Condition is met (penalty applies) if:
        # - Ketal protection is used AND
        # - No deprotection step is found AND
        # - Deprotection is required (config setting)
        condition_met = (has_ketal_protection and 
                        not has_ketal_deprotection and 
                        self.deprotection_required)
        
        return condition_met, len(reactions)
    
    def detect_ketal_protection(self, rxn):
        """Detect ketal protection reactions (ketone -> ketal)"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[0].split(".") if r.strip()]
        products = [Chem.MolFromSmiles(p.strip()) for p in rxn_parts[1].split(".") if p.strip()]
        
        if not all(reactants) or not all(products):
            return False
        
        # Ketone pattern
        ketone_pattern = Chem.MolFromSmarts("[C;!$(C=O-O)]=[O;!$(O-C=O)]")
        # Ketal pattern (carbon bonded to two oxygens, not carbonyl)
        ketal_pattern = Chem.MolFromSmarts("[C;X4](-[O;!$(O=C)])(-[O;!$(O=C)])")
        
        # Check if reactants contain ketone and products contain ketal
        reactant_has_ketone = any(mol.HasSubstructMatch(ketone_pattern) for mol in reactants)
        product_has_ketal = any(mol.HasSubstructMatch(ketal_pattern) for mol in products)
        
        return reactant_has_ketone and product_has_ketal
    
    def detect_ketal_deprotection(self, rxn):
        """Detect ketal deprotection reactions (ketal -> ketone)"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[0].split(".") if r.strip()]
        products = [Chem.MolFromSmiles(p.strip()) for p in rxn_parts[1].split(".") if p.strip()]
        
        if not all(reactants) or not all(products):
            return False
        
        # Ketal pattern (carbon bonded to two oxygens, not carbonyl)
        ketal_pattern = Chem.MolFromSmarts("[C;X4](-[O;!$(O=C)])(-[O;!$(O=C)])")
        # Ketone pattern
        ketone_pattern = Chem.MolFromSmarts("[C;!$(C=O-O)]=[O;!$(O-C=O)]")
        
        # Check if reactants contain ketal and products contain ketone
        reactant_has_ketal = any(mol.HasSubstructMatch(ketal_pattern) for mol in reactants)
        product_has_ketone = any(mol.HasSubstructMatch(ketone_pattern) for mol in products)
        
        return reactant_has_ketal and product_has_ketone
