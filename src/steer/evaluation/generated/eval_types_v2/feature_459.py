"""Generated evaluation code for: Sequential ester hydrolysis followed by re-esterification"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SequentialEsterReactions(MultiRxnCondBase):
    """
    Checks for sequential ester hydrolysis followed by re-esterification.
    Detects when a route contains consecutive ester hydrolysis and esterification reactions.
    """
    
    def __init__(self, config):
        self.require_consecutive = config.get("sequence") == "consecutive"
        self.functional_group = config.get("functional_group", "ester")
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Find positions of hydrolysis and esterification reactions
        hydrolysis_positions = []
        esterification_positions = []
        
        for i, rxn in enumerate(reactions):
            if self.detect_ester_hydrolysis(rxn):
                hydrolysis_positions.append(i)
            if self.detect_esterification(rxn):
                esterification_positions.append(i)
        
        # Check if we have both types of reactions
        has_both = len(hydrolysis_positions) > 0 and len(esterification_positions) > 0
        
        if self.require_consecutive and has_both:
            # Check for consecutive occurrence (hydrolysis followed by esterification)
            condition = self.has_consecutive_sequence(hydrolysis_positions, esterification_positions)
        else:
            condition = has_both
            
        return condition, len(reactions)
    
    def has_consecutive_sequence(self, hydrolysis_pos, esterification_pos):
        """Check if any hydrolysis is immediately followed by esterification"""
        for h_pos in hydrolysis_pos:
            if (h_pos + 1) in esterification_pos:
                return True
        return False
    
    def detect_ester_hydrolysis(self, rxn):
        """Detect ester hydrolysis: ester + H2O -> carboxylic acid + alcohol"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0].split(".")
        products = rxn_parts[1].split(".")
        
        # Look for ester pattern in reactants
        ester_pattern = Chem.MolFromSmarts("[C](=[O])[O][C]")
        carboxylic_acid_pattern = Chem.MolFromSmarts("[C](=[O])[OH]")
        water_pattern = Chem.MolFromSmarts("O")
        
        has_ester_reactant = False
        has_water_reactant = False
        has_acid_product = False
        
        # Check reactants
        for reactant_smiles in reactants:
            try:
                mol = Chem.MolFromSmiles(reactant_smiles)
                if mol:
                    if mol.HasSubstructMatch(ester_pattern):
                        has_ester_reactant = True
                    if reactant_smiles.strip() == "O":  # Water
                        has_water_reactant = True
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
        
        return has_ester_reactant and has_water_reactant and has_acid_product
    
    def detect_esterification(self, rxn):
        """Detect esterification: carboxylic acid + alcohol -> ester + H2O"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0].split(".")
        products = rxn_parts[1].split(".")
        
        carboxylic_acid_pattern = Chem.MolFromSmarts("[C](=[O])[OH]")
        alcohol_pattern = Chem.MolFromSmarts("[C][OH]")
        ester_pattern = Chem.MolFromSmarts("[C](=[O])[O][C]")
        
        has_acid_reactant = False
        has_alcohol_reactant = False
        has_ester_product = False
        has_water_product = False
        
        # Check reactants
        for reactant_smiles in reactants:
            try:
                mol = Chem.MolFromSmiles(reactant_smiles)
                if mol:
                    if mol.HasSubstructMatch(carboxylic_acid_pattern):
                        has_acid_reactant = True
                    if mol.HasSubstructMatch(alcohol_pattern):
                        has_alcohol_reactant = True
            except:
                continue
        
        # Check products
        for product_smiles in products:
            try:
                mol = Chem.MolFromSmiles(product_smiles)
                if mol:
                    if mol.HasSubstructMatch(ester_pattern):
                        has_ester_product = True
                elif product_smiles.strip() == "O":  # Water
                    has_water_product = True
            except:
                continue
        
        return has_acid_reactant and has_alcohol_reactant and has_ester_product and has_water_product
