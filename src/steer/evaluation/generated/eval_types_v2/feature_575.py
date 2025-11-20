"""Generated evaluation code for: Ester formation followed by immediate hydrolysis"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EsterFormationHydrolysis(MultiRxnCondBase):
    """
    Detects routes that contain ester formation followed by immediate hydrolysis,
    which represents a redundant reaction sequence that returns to the original
    carboxylic acid functionality.
    """
    
    def __init__(self, config):
        self.consecutive = config.get("consecutive", True)
        self.functional_group = config.get("functional_group", "carboxylic_acid")
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Find esterification and hydrolysis reactions
        ester_indices = []
        hydrolysis_indices = []
        
        for i, rxn in enumerate(reactions):
            if self.detect_esterification(rxn):
                ester_indices.append(i)
            if self.detect_hydrolysis(rxn):
                hydrolysis_indices.append(i)
        
        # Check if we have both reaction types
        has_both = len(ester_indices) > 0 and len(hydrolysis_indices) > 0
        
        if self.consecutive:
            # Check for consecutive ester formation followed by hydrolysis
            condition = self.has_consecutive_sequence(ester_indices, hydrolysis_indices)
        else:
            # Just check for presence of both reactions in the route
            condition = has_both
            
        return condition, len(reactions)
    
    def has_consecutive_sequence(self, ester_indices, hydrolysis_indices):
        """Check if esterification is immediately followed by hydrolysis"""
        for ester_idx in ester_indices:
            if (ester_idx + 1) in hydrolysis_indices:
                return True
        return False
    
    def detect_esterification(self, rxn):
        """Detect ester formation reactions"""
        # Check for carboxylic acid + alcohol -> ester + water pattern
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0].split(".")
        products = rxn_parts[1].split(".")
        
        # SMARTS patterns
        carboxylic_acid_pattern = Chem.MolFromSmarts("[CX3](=O)[OH]")
        alcohol_pattern = Chem.MolFromSmarts("[CX4][OH]")
        ester_pattern = Chem.MolFromSmarts("[CX3](=O)[OX2][CX4]")
        
        # Check reactants for carboxylic acid and alcohol
        has_acid = False
        has_alcohol = False
        for reactant_smiles in reactants:
            try:
                mol = Chem.MolFromSmiles(reactant_smiles)
                if mol:
                    if carboxylic_acid_pattern and mol.HasSubstructMatch(carboxylic_acid_pattern):
                        has_acid = True
                    if alcohol_pattern and mol.HasSubstructMatch(alcohol_pattern):
                        has_alcohol = True
            except:
                continue
        
        # Check products for ester
        has_ester = False
        for product_smiles in products:
            try:
                mol = Chem.MolFromSmiles(product_smiles)
                if mol and ester_pattern:
                    if mol.HasSubstructMatch(ester_pattern):
                        has_ester = True
                        break
            except:
                continue
        
        return has_acid and has_alcohol and has_ester
    
    def detect_hydrolysis(self, rxn):
        """Detect ester hydrolysis reactions"""
        # Check for ester + water -> carboxylic acid + alcohol pattern
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0].split(".")
        products = rxn_parts[1].split(".")
        
        # SMARTS patterns
        ester_pattern = Chem.MolFromSmarts("[CX3](=O)[OX2][CX4]")
        carboxylic_acid_pattern = Chem.MolFromSmarts("[CX3](=O)[OH]")
        alcohol_pattern = Chem.MolFromSmarts("[CX4][OH]")
        
        # Check reactants for ester
        has_ester = False
        for reactant_smiles in reactants:
            try:
                mol = Chem.MolFromSmiles(reactant_smiles)
                if mol and ester_pattern:
                    if mol.HasSubstructMatch(ester_pattern):
                        has_ester = True
                        break
            except:
                continue
        
        # Check products for carboxylic acid and alcohol
        has_acid = False
        has_alcohol = False
        for product_smiles in products:
            try:
                mol = Chem.MolFromSmiles(product_smiles)
                if mol:
                    if carboxylic_acid_pattern and mol.HasSubstructMatch(carboxylic_acid_pattern):
                        has_acid = True
                    if alcohol_pattern and mol.HasSubstructMatch(alcohol_pattern):
                        has_alcohol = True
            except:
                continue
        
        return has_ester and has_acid and has_alcohol
