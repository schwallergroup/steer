"""Generated evaluation code for: Ketone reduction after acid chloride formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class KetoneReductionAfterAcidChloride(MultiRxnCondBase):
    """
    Checks for the problematic sequence of acid chloride formation followed by ketone reduction.
    This represents a selectivity issue since acid chlorides are more reactive than ketones
    toward reducing agents, making selective ketone reduction challenging.
    """
    
    def __init__(self, config):
        self.penalize_sequence = config.get("penalize_sequence", True)
        self.acid_chloride_pattern = Chem.MolFromSmarts("[CX3](=[OX1])[Cl]")  # Acid chloride
        self.ketone_pattern = Chem.MolFromSmarts("[CX3](=[OX1])([C,c])[C,c]")  # Ketone
        self.carboxylic_acid_pattern = Chem.MolFromSmarts("[CX3](=[OX1])[OX2H1]")  # Carboxylic acid
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Track the sequence of reactions
        acid_chloride_step = -1
        ketone_reduction_step = -1
        
        for i, rxn in enumerate(reactions):
            if self.detect_acid_chloride_formation(rxn):
                acid_chloride_step = i
            elif self.detect_ketone_reduction(rxn):
                ketone_reduction_step = i
        
        # Check if acid chloride formation occurs before ketone reduction
        problematic_sequence = (acid_chloride_step >= 0 and 
                              ketone_reduction_step >= 0 and 
                              acid_chloride_step < ketone_reduction_step)
        
        if self.penalize_sequence:
            condition = not problematic_sequence  # Penalize the bad sequence
        else:
            condition = problematic_sequence  # Reward finding the sequence
            
        return condition, len(reactions)
    
    def detect_acid_chloride_formation(self, rxn):
        """Detect conversion of carboxylic acid to acid chloride"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        try:
            # Check reactants for carboxylic acid
            reactant_mols = [Chem.MolFromSmiles(smi.strip()) 
                           for smi in reactants.split(".") if smi.strip()]
            has_carboxylic_acid = any(mol and mol.HasSubstructMatch(self.carboxylic_acid_pattern) 
                                    for mol in reactant_mols if mol)
            
            # Check products for acid chloride
            product_mols = [Chem.MolFromSmiles(smi.strip()) 
                          for smi in products.split(".") if smi.strip()]
            has_acid_chloride = any(mol and mol.HasSubstructMatch(self.acid_chloride_pattern) 
                                  for mol in product_mols if mol)
            
            return has_carboxylic_acid and has_acid_chloride
            
        except:
            return False
    
    def detect_ketone_reduction(self, rxn):
        """Detect reduction of ketone to alcohol"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        try:
            # Check reactants for ketone
            reactant_mols = [Chem.MolFromSmiles(smi.strip()) 
                           for smi in reactants.split(".") if smi.strip()]
            has_ketone = any(mol and mol.HasSubstructMatch(self.ketone_pattern) 
                           for mol in reactant_mols if mol)
            
            # Check products for secondary alcohol (ketone reduction product)
            alcohol_pattern = Chem.MolFromSmarts("[CX4H1]([C,c])([C,c])[OH1]")
            product_mols = [Chem.MolFromSmiles(smi.strip()) 
                          for smi in products.split(".") if smi.strip()]
            has_alcohol = any(mol and mol.HasSubstructMatch(alcohol_pattern) 
                            for mol in product_mols if mol)
            
            return has_ketone and has_alcohol
            
        except:
            return False
