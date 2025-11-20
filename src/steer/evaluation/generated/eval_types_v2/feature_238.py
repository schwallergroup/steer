"""Generated evaluation code for: Benzyl protecting group strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BenzylProtectingGroupStrategy(MultiRxnCondBase):
    """
    Evaluates benzyl protecting group strategy in synthesis routes.
    Checks for presence of N-benzyl and O-benzyl protecting groups and
    their simultaneous removal via hydrogenation reactions.
    """
    
    def __init__(self, config):
        self.protecting_groups = config.get("protecting_groups", ["N-benzyl", "O-benzyl"])
        self.deprotection_method = config.get("deprotection_method", "hydrogenation")
        self.simultaneous_removal = config.get("simultaneous_removal", True)
        
        # SMARTS patterns for protecting groups
        self.nbenzyl_pattern = "[NH1,NH2][CH2]c1ccccc1"  # N-benzyl
        self.obenzyl_pattern = "[OH0][CH2]c1ccccc1"      # O-benzyl
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        has_nbenzyl_protection = False
        has_obenzyl_protection = False
        has_hydrogenation_deprotection = False
        simultaneous_deprotection = False
        
        # Check each reaction for protection/deprotection
        for rxn in reactions:
            # Check for protection installation
            if "N-benzyl" in self.protecting_groups and self.detect_nbenzyl_protection(rxn):
                has_nbenzyl_protection = True
            if "O-benzyl" in self.protecting_groups and self.detect_obenzyl_protection(rxn):
                has_obenzyl_protection = True
                
            # Check for hydrogenation deprotection
            if self.detect_hydrogenation_deprotection(rxn):
                has_hydrogenation_deprotection = True
                
                # Check if both groups are removed simultaneously
                if self.simultaneous_removal and self.detect_simultaneous_removal(rxn):
                    simultaneous_deprotection = True
        
        # Evaluate overall condition
        protection_met = True
        if "N-benzyl" in self.protecting_groups:
            protection_met &= has_nbenzyl_protection
        if "O-benzyl" in self.protecting_groups:
            protection_met &= has_obenzyl_protection
            
        deprotection_met = has_hydrogenation_deprotection
        if self.simultaneous_removal and len(self.protecting_groups) > 1:
            deprotection_met &= simultaneous_deprotection
            
        condition = protection_met and deprotection_met
        return condition, len(reactions)
    
    def detect_nbenzyl_protection(self, rxn):
        """Detect N-benzyl protection installation"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[0].split(".") if r.strip()]
        products = [Chem.MolFromSmiles(p.strip()) for p in rxn_parts[1].split(".") if p.strip()]
        
        # Check if N-benzyl group is formed (not present in reactants but present in products)
        reactant_has_nbenzyl = any(mol and mol.HasSubstructMatch(Chem.MolFromSmarts(self.nbenzyl_pattern)) 
                                  for mol in reactants if mol)
        product_has_nbenzyl = any(mol and mol.HasSubstructMatch(Chem.MolFromSmarts(self.nbenzyl_pattern)) 
                                 for mol in products if mol)
        
        return not reactant_has_nbenzyl and product_has_nbenzyl
    
    def detect_obenzyl_protection(self, rxn):
        """Detect O-benzyl protection installation"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[0].split(".") if r.strip()]
        products = [Chem.MolFromSmiles(p.strip()) for p in rxn_parts[1].split(".") if p.strip()]
        
        # Check if O-benzyl group is formed
        reactant_has_obenzyl = any(mol and mol.HasSubstructMatch(Chem.MolFromSmarts(self.obenzyl_pattern)) 
                                  for mol in reactants if mol)
        product_has_obenzyl = any(mol and mol.HasSubstructMatch(Chem.MolFromSmarts(self.obenzyl_pattern)) 
                                 for mol in products if mol)
        
        return not reactant_has_obenzyl and product_has_obenzyl
    
    def detect_hydrogenation_deprotection(self, rxn):
        """Detect hydrogenation-based deprotection"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[0].split(".") if r.strip()]
        products = [Chem.MolFromSmiles(p.strip()) for p in rxn_parts[1].split(".") if p.strip()]
        
        # Check for presence of H2 in reactants (common in hydrogenation)
        has_hydrogen = any(r.strip() == "[H][H]" or r.strip() == "H2" for r in rxn_parts[0].split("."))
        
        # Check if benzyl groups are removed
        reactant_has_benzyl = any(mol and (mol.HasSubstructMatch(Chem.MolFromSmarts(self.nbenzyl_pattern)) or 
                                          mol.HasSubstructMatch(Chem.MolFromSmarts(self.obenzyl_pattern)))
                                 for mol in reactants if mol)
        product_has_benzyl = any(mol and (mol.HasSubstructMatch(Chem.MolFromSmarts(self.nbenzyl_pattern)) or 
                                         mol.HasSubstructMatch(Chem.MolFromSmarts(self.obenzyl_pattern)))
                                for mol in products if mol)
        
        return (has_hydrogen or self.has_hydrogenation_conditions(rxn)) and reactant_has_benzyl and not product_has_benzyl
    
    def detect_simultaneous_removal(self, rxn):
        """Detect simultaneous removal of both N-benzyl and O-benzyl groups"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[0].split(".") if r.strip()]
        products = [Chem.MolFromSmiles(p.strip()) for p in rxn_parts[1].split(".") if p.strip()]
        
        # Check if both types are present in reactants but absent in products
        reactant_has_nbenzyl = any(mol and mol.HasSubstructMatch(Chem.MolFromSmarts(self.nbenzyl_pattern)) 
                                  for mol in reactants if mol)
        reactant_has_obenzyl = any(mol and mol.HasSubstructMatch(Chem.MolFromSmarts(self.obenzyl_pattern)) 
                                  for mol in reactants if mol)
        
        product_has_nbenzyl = any(mol and mol.HasSubstructMatch(Chem.MolFromSmarts(self.nbenzyl_pattern)) 
                                 for mol in products if mol)
        product_has_obenzyl = any(mol and mol.HasSubstructMatc
