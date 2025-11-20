"""Generated evaluation code for: Separate benzyl protecting group removal steps"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SeparateBenzylDeprotection(MultiRxnCondBase):
    """
    Evaluates whether benzyl protecting group removal occurs in separate steps
    rather than simultaneously. Checks for N-benzyl and O-benzyl deprotection
    reactions and ensures they happen in different reaction steps.
    """
    
    def __init__(self, config):
        self.protecting_group = config.get("protecting_group", "benzyl")
        self.removal_steps = config.get("removal_steps", 2)
        self.simultaneous_removal = config.get("simultaneous_removal", False)
        
        # SMARTS patterns for benzyl groups
        self.n_benzyl_pattern = Chem.MolFromSmarts("[#7]-[CH2]-c1ccccc1")  # N-benzyl
        self.o_benzyl_pattern = Chem.MolFromSmarts("[#8]-[CH2]-c1ccccc1")  # O-benzyl
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Find reactions that involve benzyl deprotection
        n_benzyl_reactions = []
        o_benzyl_reactions = []
        
        for i, rxn in enumerate(reactions):
            if self.is_n_benzyl_deprotection(rxn):
                n_benzyl_reactions.append(i)
            if self.is_o_benzyl_deprotection(rxn):
                o_benzyl_reactions.append(i)
        
        # Check if we have both types of deprotections
        has_both_types = len(n_benzyl_reactions) > 0 and len(o_benzyl_reactions) > 0
        
        if not has_both_types:
            # If we don't have both types, condition is not applicable
            condition = True
        else:
            # Check if deprotections occur in separate steps
            if self.simultaneous_removal:
                # Want simultaneous removal - check if any reaction does both
                condition = any(self.is_n_benzyl_deprotection(rxn) and self.is_o_benzyl_deprotection(rxn) 
                              for rxn in reactions)
            else:
                # Want separate removal - check that no single reaction does both
                condition = not any(self.is_n_benzyl_deprotection(rxn) and self.is_o_benzyl_deprotection(rxn) 
                                  for rxn in reactions)
        
        return condition, len(reactions)
    
    def is_n_benzyl_deprotection(self, rxn):
        """Check if reaction involves N-benzyl deprotection"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        # Check if N-benzyl is present in reactants but not products
        reactant_mols = [Chem.MolFromSmiles(smi.strip()) for smi in reactants.split(".") if smi.strip()]
        product_mols = [Chem.MolFromSmiles(smi.strip()) for smi in products.split(".") if smi.strip()]
        
        reactant_has_n_benzyl = any(mol and mol.HasSubstructMatch(self.n_benzyl_pattern) 
                                   for mol in reactant_mols if mol)
        product_has_n_benzyl = any(mol and mol.HasSubstructMatch(self.n_benzyl_pattern) 
                                  for mol in product_mols if mol)
        
        return reactant_has_n_benzyl and not product_has_n_benzyl
    
    def is_o_benzyl_deprotection(self, rxn):
        """Check if reaction involves O-benzyl deprotection"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        # Check if O-benzyl is present in reactants but not products
        reactant_mols = [Chem.MolFromSmiles(smi.strip()) for smi in reactants.split(".") if smi.strip()]
        product_mols = [Chem.MolFromSmiles(smi.strip()) for smi in products.split(".") if smi.strip()]
        
        reactant_has_o_benzyl = any(mol and mol.HasSubstructMatch(self.o_benzyl_pattern) 
                                   for mol in reactant_mols if mol)
        product_has_o_benzyl = any(mol and mol.HasSubstructMatch(self.o_benzyl_pattern) 
                                  for mol in product_mols if mol)
        
        return reactant_has_o_benzyl and not product_has_o_benzyl
