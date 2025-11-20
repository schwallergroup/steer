"""Generated evaluation code for: Multiple esterification hydrolysis cycles"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class MultipleEsterificationHydrolysisCycles(MultiRxnCondBase):
    """
    Checks for multiple sequential esterification-hydrolysis cycles in a synthesis route.
    Detects patterns where esters are formed and then hydrolyzed repeatedly.
    """
    
    def __init__(self, config):
        self.cycle_count = config.get("cycle_count", 3)
        self.sequential = config.get("sequential", True)
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Classify each reaction as esterification or hydrolysis
        reaction_types = []
        for rxn in reactions:
            if self.detect_esterification(rxn):
                reaction_types.append("esterification")
            elif self.detect_hydrolysis(rxn):
                reaction_types.append("hydrolysis")
            else:
                reaction_types.append("other")
        
        if self.sequential:
            cycles_found = self.count_sequential_cycles(reaction_types)
        else:
            cycles_found = self.count_any_cycles(reaction_types)
        
        condition = cycles_found >= self.cycle_count
        return condition, len(reactions)
    
    def detect_esterification(self, rxn):
        """Detect esterification reaction (carboxylic acid + alcohol -> ester + water)"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0].split(".")
        products = rxn_parts[1].split(".")
        
        # Look for carboxylic acid pattern in reactants and ester pattern in products
        carboxylic_acid_pattern = Chem.MolFromSmarts("[CX3](=O)[OH]")
        alcohol_pattern = Chem.MolFromSmarts("[CX4][OH]")
        ester_pattern = Chem.MolFromSmarts("[CX3](=O)[OX2][CX4]")
        
        has_acid = False
        has_alcohol = False
        has_ester = False
        
        # Check reactants for acid and alcohol
        for reactant_smiles in reactants:
            try:
                mol = Chem.MolFromSmiles(reactant_smiles)
                if mol and mol.HasSubstructMatch(carboxylic_acid_pattern):
                    has_acid = True
                if mol and mol.HasSubstructMatch(alcohol_pattern):
                    has_alcohol = True
            except:
                continue
        
        # Check products for ester
        for product_smiles in products:
            try:
                mol = Chem.MolFromSmiles(product_smiles)
                if mol and mol.HasSubstructMatch(ester_pattern):
                    has_ester = True
            except:
                continue
        
        return has_acid and has_alcohol and has_ester
    
    def detect_hydrolysis(self, rxn):
        """Detect hydrolysis reaction (ester + water -> carboxylic acid + alcohol)"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0].split(".")
        products = rxn_parts[1].split(".")
        
        # Look for ester pattern in reactants and carboxylic acid in products
        ester_pattern = Chem.MolFromSmarts("[CX3](=O)[OX2][CX4]")
        carboxylic_acid_pattern = Chem.MolFromSmarts("[CX3](=O)[OH]")
        
        has_ester = False
        has_acid_product = False
        
        # Check reactants for ester
        for reactant_smiles in reactants:
            try:
                mol = Chem.MolFromSmiles(reactant_smiles)
                if mol and mol.HasSubstructMatch(ester_pattern):
                    has_ester = True
            except:
                continue
        
        # Check products for carboxylic acid
        for product_smiles in products:
            try:
                mol = Chem.MolFromSmiles(product_smiles)
                if mol and mol.HasSubstructMatch(carboxylic_acid_pattern):
                    has_acid_product = True
            except:
                continue
        
        return has_ester and has_acid_product
    
    def count_sequential_cycles(self, reaction_types):
        """Count sequential esterification-hydrolysis cycles"""
        cycles = 0
        i = 0
        while i < len(reaction_types) - 1:
            if (reaction_types[i] == "esterification" and 
                reaction_types[i + 1] == "hydrolysis"):
                cycles += 1
                i += 2  # Skip both reactions in the cycle
            else:
                i += 1
        return cycles
    
    def count_any_cycles(self, reaction_types):
        """Count total pairs of esterification and hydrolysis reactions (not necessarily sequential)"""
        esterification_count = reaction_types.count("esterification")
        hydrolysis_count = reaction_types.count("hydrolysis")
        return min(esterification_count, hydrolysis_count)
