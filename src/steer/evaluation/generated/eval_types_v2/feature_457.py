"""Generated evaluation code for: Alcohol protection before nitro reduction"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class AlcoholProtectionBeforeNitroReduction(MultiRxnCondBase):
    """
    Checks if alcohol protection (acetylation) occurs before nitro reduction in the synthesis route.
    Evaluates the sequential order of these transformations to ensure proper protecting group strategy.
    """
    
    def __init__(self, config):
        self.require_proper_sequence = config.get("require_proper_sequence", True)
        self.alcohol_pattern = "[OH1][CH2,CH1,CH0]"  # Primary, secondary, tertiary alcohols
        self.acetate_pattern = "CC(=O)O[CH2,CH1,CH0]"  # Acetate protected alcohol
        self.nitro_pattern = "[N+](=O)[O-]"  # Nitro group
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        acetylation_depth = -1
        nitro_reduction_depth = -1
        
        # Find depths of acetylation and nitro reduction reactions
        for i, rxn in enumerate(reactions):
            if self.detect_acetylation(rxn):
                acetylation_depth = i
            if self.detect_nitro_reduction(rxn):
                nitro_reduction_depth = i
        
        # Check if both reactions occur and acetylation comes before nitro reduction
        if acetylation_depth >= 0 and nitro_reduction_depth >= 0:
            condition = acetylation_depth < nitro_reduction_depth
        else:
            # If one or both reactions don't occur, condition is not met
            condition = False
            
        return condition, len(reactions)
    
    def detect_acetylation(self, rxn):
        """Detects alcohol acetylation reaction"""
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants = rxn_parts[0].split(".")
            products = rxn_parts[1].split(".")
            
            # Check for alcohol in reactants and acetate in products
            has_alcohol_reactant = False
            has_acetate_product = False
            
            for r_smiles in reactants:
                mol = Chem.MolFromSmiles(r_smiles)
                if mol and mol.HasSubstructMatch(Chem.MolFromSmarts(self.alcohol_pattern)):
                    has_alcohol_reactant = True
                    break
            
            for p_smiles in products:
                mol = Chem.MolFromSmiles(p_smiles)
                if mol and mol.HasSubstructMatch(Chem.MolFromSmarts(self.acetate_pattern)):
                    has_acetate_product = True
                    break
            
            return has_alcohol_reactant and has_acetate_product
            
        except Exception:
            return False
    
    def detect_nitro_reduction(self, rxn):
        """Detects nitro group reduction reaction"""
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants = rxn_parts[0].split(".")
            products = rxn_parts[1].split(".")
            
            # Check for nitro group in reactants and absence in products (or amine formation)
            has_nitro_reactant = False
            nitro_reduced_in_products = False
            
            for r_smiles in reactants:
                mol = Chem.MolFromSmiles(r_smiles)
                if mol and mol.HasSubstructMatch(Chem.MolFromSmarts(self.nitro_pattern)):
                    has_nitro_reactant = True
                    break
            
            if has_nitro_reactant:
                # Check if nitro is reduced (less nitro groups in products or amine formed)
                reactant_nitro_count = sum(
                    len(Chem.MolFromSmiles(r).GetSubstructMatches(Chem.MolFromSmarts(self.nitro_pattern)))
                    for r in reactants if Chem.MolFromSmiles(r)
                )
                product_nitro_count = sum(
                    len(Chem.MolFromSmiles(p).GetSubstructMatches(Chem.MolFromSmarts(self.nitro_pattern)))
                    for p in products if Chem.MolFromSmiles(p)
                )
                nitro_reduced_in_products = product_nitro_count < reactant_nitro_count
            
            return has_nitro_reactant and nitro_reduced_in_products
            
        except Exception:
            return False
