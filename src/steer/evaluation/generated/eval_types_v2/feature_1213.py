"""Generated evaluation code for: Methyl ester protection for compatible oxidation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class MethylEsterProtection(MultiRxnCondBase):
    """
    Evaluates synthesis routes for methyl ester protection strategy.
    Checks if carboxylic acid is protected as methyl ester before alcohol oxidation.
    """
    
    def __init__(self, config):
        self.protection_type = config.get("protection_type", "methyl_ester")
        self.protected_group = config.get("protected_group", "carboxylic_acid")
        self.intervening_reaction = config.get("intervening_reaction", "alcohol_oxidation")
        self.require_protection = config.get("require_protection", True)

    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Find alcohol oxidation reactions
        oxidation_indices = []
        for i, rxn in enumerate(reactions):
            if self.detect_alcohol_oxidation(rxn):
                oxidation_indices.append(i)
        
        if not oxidation_indices:
            # No alcohol oxidation found - condition met if protection not required
            return not self.require_protection, len(reactions)
        
        # Check if methyl ester protection occurs before any alcohol oxidation
        protection_found = False
        for i, rxn in enumerate(reactions):
            if self.detect_methyl_ester_formation(rxn):
                protection_found = True
                break
            elif i in oxidation_indices:
                # Hit oxidation before protection
                break
        
        condition = protection_found == self.require_protection
        return condition, len(reactions)

    def detect_alcohol_oxidation(self, rxn):
        """Detect alcohol to aldehyde/ketone oxidation reactions"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[0].split(".")]
        products = [Chem.MolFromSmiles(p.strip()) for p in rxn_parts[1].split(".")]
        
        if not all(reactants) or not all(products):
            return False
        
        # Primary alcohol pattern
        primary_alcohol = Chem.MolFromSmarts("[CH2][OH]")
        # Secondary alcohol pattern  
        secondary_alcohol = Chem.MolFromSmarts("[CH]([OH])")
        # Aldehyde pattern
        aldehyde = Chem.MolFromSmarts("[CH1](=O)")
        # Ketone pattern
        ketone = Chem.MolFromSmarts("[C](=O)")
        
        # Check if reactants contain alcohol and products contain carbonyl
        has_alcohol = any(mol.HasSubstructMatch(primary_alcohol) or 
                         mol.HasSubstructMatch(secondary_alcohol) 
                         for mol in reactants)
        has_carbonyl = any(mol.HasSubstructMatch(aldehyde) or 
                          mol.HasSubstructMatch(ketone) 
                          for mol in products)
        
        return has_alcohol and has_carbonyl

    def detect_methyl_ester_formation(self, rxn):
        """Detect carboxylic acid to methyl ester protection"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[0].split(".")]
        products = [Chem.MolFromSmiles(p.strip()) for p in rxn_parts[1].split(".")]
        
        if not all(reactants) or not all(products):
            return False
        
        # Carboxylic acid pattern
        carboxylic_acid = Chem.MolFromSmarts("[C](=O)[OH]")
        # Methyl ester pattern
        methyl_ester = Chem.MolFromSmarts("[C](=O)[O][CH3]")
        
        # Check if reactants contain carboxylic acid and products contain methyl ester
        has_acid = any(mol.HasSubstructMatch(carboxylic_acid) for mol in reactants)
        has_methyl_ester = any(mol.HasSubstructMatch(methyl_ester) for mol in products)
        
        return has_acid and has_methyl_ester

    def route_scoring(self, x):
        """Score based on whether the protection strategy is properly implemented"""
        if x < 0:
            return 0  # Strategy not found
        else:
            return 10  # Strategy successfully implemented
