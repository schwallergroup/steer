"""Generated evaluation code for: Sequential benzyl deprotection strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SequentialBenzylDeprotection(MultiRxnCondBase):
    """
    Evaluates whether a route uses sequential benzyl deprotection strategy,
    performing separate N-debenzylation and O-debenzylation steps instead 
    of simultaneous removal.
    """
    
    def __init__(self, config):
        self.substrate_types = config.get("substrate_types", ["N-benzyl", "O-benzyl"])
        self.require_sequential = config.get("require_sequential", True)
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Find all benzyl deprotection reactions
        n_debenzyl_reactions = []
        o_debenzyl_reactions = []
        
        for i, rxn in enumerate(reactions):
            if self.is_n_debenzylation(rxn):
                n_debenzyl_reactions.append(i)
            elif self.is_o_debenzylation(rxn):
                o_debenzyl_reactions.append(i)
        
        # Check if we have both types of deprotections
        has_n_debenzyl = len(n_debenzyl_reactions) > 0
        has_o_debenzyl = len(o_debenzyl_reactions) > 0
        
        if not (has_n_debenzyl and has_o_debenzyl):
            # No sequential strategy possible/needed
            condition = True
            return condition, len(reactions)
        
        # Check if deprotections are sequential (separate steps)
        is_sequential = self.are_reactions_sequential(reactions, n_debenzyl_reactions, o_debenzyl_reactions)
        
        condition = is_sequential if self.require_sequential else not is_sequential
        return condition, len(reactions)
    
    def is_n_debenzylation(self, rxn):
        """Check if reaction involves N-benzyl deprotection"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        # N-benzyl pattern (nitrogen attached to benzyl group)
        n_benzyl_pattern = Chem.MolFromSmarts("[NH0,NH1,NH2]-[CH2]-c1ccccc1")
        
        try:
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".")]
            
            # Check if reactants contain N-benzyl and products don't (or have fewer)
            reactant_n_benzyl = sum(mol.HasSubstructMatch(n_benzyl_pattern) for mol in reactant_mols if mol)
            product_n_benzyl = sum(mol.HasSubstructMatch(n_benzyl_pattern) for mol in product_mols if mol)
            
            return reactant_n_benzyl > product_n_benzyl
            
        except:
            return False
    
    def is_o_debenzylation(self, rxn):
        """Check if reaction involves O-benzyl deprotection"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        # O-benzyl pattern (oxygen attached to benzyl group)
        o_benzyl_pattern = Chem.MolFromSmarts("O-[CH2]-c1ccccc1")
        
        try:
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".")]
            
            # Check if reactants contain O-benzyl and products don't (or have fewer)
            reactant_o_benzyl = sum(mol.HasSubstructMatch(o_benzyl_pattern) for mol in reactant_mols if mol)
            product_o_benzyl = sum(mol.HasSubstructMatch(o_benzyl_pattern) for mol in product_mols if mol)
            
            return reactant_o_benzyl > product_o_benzyl
            
        except:
            return False
    
    def are_reactions_sequential(self, reactions, n_debenzyl_indices, o_debenzyl_indices):
        """
        Check if N-debenzylation and O-debenzylation occur in separate reactions
        rather than simultaneously in the same reaction
        """
        # If any reaction does both N and O debenzylation, it's not sequential
        for i in range(len(reactions)):
            if i in n_debenzyl_indices and i in o_debenzyl_indices:
                return False
        
        # If deprotections are in different reactions, it's sequential
        return True
