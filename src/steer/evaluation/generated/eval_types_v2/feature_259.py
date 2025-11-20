"""Generated evaluation code for: Multiple benzyl protecting groups strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class MultipleBenzylProtectingGroups(MultiRxnCondBase):
    """
    Evaluates synthesis routes that use multiple benzyl protecting groups simultaneously.
    Detects the presence of both N-benzyl and O-benzyl protection in the same route,
    which can create selectivity challenges during deprotection steps.
    """
    
    def __init__(self, config):
        self.require_multiple = config.get("count") == "multiple"
        self.check_selectivity = config.get("selectivity_issue", True)
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Track benzyl protection formations
        n_benzyl_reactions = []
        o_benzyl_reactions = []
        
        for i, rxn in enumerate(reactions):
            if self.detect_n_benzyl_protection(rxn):
                n_benzyl_reactions.append(i)
            if self.detect_o_benzyl_protection(rxn):
                o_benzyl_reactions.append(i)
        
        # Check if we have both types of benzyl protection
        has_n_benzyl = len(n_benzyl_reactions) > 0
        has_o_benzyl = len(o_benzyl_reactions) > 0
        has_multiple_benzyl = has_n_benzyl and has_o_benzyl
        
        if self.require_multiple:
            condition = has_multiple_benzyl
        else:
            condition = has_n_benzyl or has_o_benzyl
            
        return condition, len(reactions)
    
    def detect_n_benzyl_protection(self, rxn):
        """Detect N-benzyl protection formation (benzylation of nitrogen)"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        try:
            # Look for primary or secondary amine in reactants
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".")]
            
            # Primary amine pattern
            primary_amine_pattern = Chem.MolFromSmarts("[NH2]")
            # Secondary amine pattern  
            secondary_amine_pattern = Chem.MolFromSmarts("[NH1]")
            # N-benzyl pattern
            n_benzyl_pattern = Chem.MolFromSmarts("[NH1,NH0]-[CH2]-c1ccccc1")
            
            # Check if reactants have free amine
            has_free_amine = False
            for mol in reactant_mols:
                if mol and (mol.HasSubstructMatch(primary_amine_pattern) or 
                           mol.HasSubstructMatch(secondary_amine_pattern)):
                    has_free_amine = True
                    break
            
            # Check if products have N-benzyl group
            has_n_benzyl_product = False
            for mol in product_mols:
                if mol and mol.HasSubstructMatch(n_benzyl_pattern):
                    has_n_benzyl_product = True
                    break
                    
            return has_free_amine and has_n_benzyl_product
            
        except:
            return False
    
    def detect_o_benzyl_protection(self, rxn):
        """Detect O-benzyl protection formation (benzylation of oxygen)"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        try:
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".")]
            
            # Alcohol or phenol pattern
            alcohol_pattern = Chem.MolFromSmarts("[OH1]")
            # O-benzyl pattern
            o_benzyl_pattern = Chem.MolFromSmarts("O-[CH2]-c1ccccc1")
            
            # Check if reactants have free alcohol/phenol
            has_free_alcohol = False
            for mol in reactant_mols:
                if mol and mol.HasSubstructMatch(alcohol_pattern):
                    has_free_alcohol = True
                    break
            
            # Check if products have O-benzyl group
            has_o_benzyl_product = False
            for mol in product_mols:
                if mol and mol.HasSubstructMatch(o_benzyl_pattern):
                    has_o_benzyl_product = True
                    break
                    
            return has_free_alcohol and has_o_benzyl_product
            
        except:
            return False
