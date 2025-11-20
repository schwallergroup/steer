"""Generated evaluation code for: Two-step sulfide oxidation strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class TwoStepSulfideOxidation(MultiRxnCondBase):
    """
    Evaluates whether a synthesis route uses a controlled two-step sulfide oxidation strategy.
    Checks for sequential oxidation from sulfide (R-S-R) to sulfoxide (R-SO-R) to sulfone (R-SO2-R).
    """
    
    def __init__(self, config):
        self.require_sequential = config.get("require_sequential", True)
        self.allow_direct_oxidation = config.get("allow_direct_oxidation", False)
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Find all sulfide oxidation reactions
        sulfide_to_sulfoxide_rxns = []
        sulfoxide_to_sulfone_rxns = []
        direct_sulfide_to_sulfone_rxns = []
        
        for i, rxn in enumerate(reactions):
            if self.is_sulfide_to_sulfoxide(rxn):
                sulfide_to_sulfoxide_rxns.append(i)
            elif self.is_sulfoxide_to_sulfone(rxn):
                sulfoxide_to_sulfone_rxns.append(i)
            elif self.is_direct_sulfide_to_sulfone(rxn):
                direct_sulfide_to_sulfone_rxns.append(i)
        
        # Check for two-step sequential oxidation
        has_two_step = len(sulfide_to_sulfoxide_rxns) > 0 and len(sulfoxide_to_sulfone_rxns) > 0
        has_direct = len(direct_sulfide_to_sulfone_rxns) > 0
        
        if self.require_sequential:
            condition = has_two_step and (not has_direct or self.allow_direct_oxidation)
        else:
            condition = has_two_step or has_direct
        
        total_oxidation_steps = len(sulfide_to_sulfoxide_rxns) + len(sulfoxide_to_sulfone_rxns) + len(direct_sulfide_to_sulfone_rxns)
        
        return condition, total_oxidation_steps
    
    def is_sulfide_to_sulfoxide(self, rxn_smiles):
        """Check if reaction converts sulfide to sulfoxide"""
        try:
            reactants, products = rxn_smiles.split(">>")
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".")]
            
            # Patterns for sulfide and sulfoxide
            sulfide_pattern = Chem.MolFromSmarts("[#16X2]([#6])[#6]")  # R-S-R
            sulfoxide_pattern = Chem.MolFromSmarts("[#16X3](=[#8])([#6])[#6]")  # R-SO-R
            
            # Check if reactants contain sulfide and products contain sulfoxide
            has_sulfide_reactant = any(mol and mol.HasSubstructMatch(sulfide_pattern) for mol in reactant_mols if mol)
            has_sulfoxide_product = any(mol and mol.HasSubstructMatch(sulfoxide_pattern) for mol in product_mols if mol)
            
            return has_sulfide_reactant and has_sulfoxide_product
        except:
            return False
    
    def is_sulfoxide_to_sulfone(self, rxn_smiles):
        """Check if reaction converts sulfoxide to sulfone"""
        try:
            reactants, products = rxn_smiles.split(">>")
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".")]
            
            # Patterns for sulfoxide and sulfone
            sulfoxide_pattern = Chem.MolFromSmarts("[#16X3](=[#8])([#6])[#6]")  # R-SO-R
            sulfone_pattern = Chem.MolFromSmarts("[#16X4](=[#8])(=[#8])([#6])[#6]")  # R-SO2-R
            
            # Check if reactants contain sulfoxide and products contain sulfone
            has_sulfoxide_reactant = any(mol and mol.HasSubstructMatch(sulfoxide_pattern) for mol in reactant_mols if mol)
            has_sulfone_product = any(mol and mol.HasSubstructMatch(sulfone_pattern) for mol in product_mols if mol)
            
            return has_sulfoxide_reactant and has_sulfone_product
        except:
            return False
    
    def is_direct_sulfide_to_sulfone(self, rxn_smiles):
        """Check if reaction directly converts sulfide to sulfone (single step)"""
        try:
            reactants, products = rxn_smiles.split(">>")
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".")]
            
            # Patterns for sulfide and sulfone
            sulfide_pattern = Chem.MolFromSmarts("[#16X2]([#6])[#6]")  # R-S-R
            sulfone_pattern = Chem.MolFromSmarts("[#16X4](=[#8])(=[#8])([#6])[#6]")  # R-SO2-R
            
            # Check if reactants contain sulfide and products contain sulfone
            has_sulfide_reactant = any(mol and mol.HasSubstructMatch(sulfide_pattern) for mol in reactant_mols if mol)
            has_sulfone_product = any(mol and mol.HasSubstructMatch(sulfone_pattern) for mol in product_mols if mol)
            
            return has_sulfide_reactant and has_sulfone_product
        except:
            return False
