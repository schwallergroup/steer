"""Generated evaluation code for: Ester hydrolysis followed by re-esterification sequence"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EsterHydrolysisReesterification(MultiRxnCondBase):
    """
    Evaluates synthesis routes for the presence of ester hydrolysis followed by re-esterification sequence.
    Checks if the route contains ethyl ester hydrolysis immediately followed by methyl ester formation.
    """
    
    def __init__(self, config):
        self.consecutive = config.get("consecutive", True)
        self.ester_hydrolysis_pattern = "[C:1](=[O:2])[O:3][CH2:4][CH3:5]>>[C:1](=[O:2])[OH:3]"  # Ethyl ester to carboxylic acid
        self.esterification_pattern = "[C:1](=[O:2])[OH:3]>>[C:1](=[O:2])[O:3][CH3:4]"  # Carboxylic acid to methyl ester
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        if self.consecutive:
            # Check for consecutive sequence
            for i in range(len(reactions) - 1):
                if (self.detect_ester_hydrolysis(reactions[i]) and 
                    self.detect_esterification(reactions[i + 1])):
                    return True, len(reactions)
        else:
            # Check for presence of both reactions anywhere in the route
            has_hydrolysis = any(self.detect_ester_hydrolysis(r) for r in reactions)
            has_esterification = any(self.detect_esterification(r) for r in reactions)
            if has_hydrolysis and has_esterification:
                return True, len(reactions)
        
        return False, len(reactions)
    
    def detect_ester_hydrolysis(self, rxn):
        """Detect ethyl ester hydrolysis to carboxylic acid"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0]
        products = rxn_parts[1].split(".")
        
        # Check for ethyl ester pattern in reactants
        ethyl_ester_pattern = Chem.MolFromSmarts("C(=O)OCC")
        carboxylic_acid_pattern = Chem.MolFromSmarts("C(=O)O")
        
        try:
            reactant_mol = Chem.MolFromSmiles(reactants)
            if not reactant_mol:
                return False
                
            has_ethyl_ester = reactant_mol.HasSubstructMatch(ethyl_ester_pattern)
            
            # Check for carboxylic acid in products
            has_carboxylic_acid = False
            for product_smiles in products:
                product_mol = Chem.MolFromSmiles(product_smiles)
                if product_mol and product_mol.HasSubstructMatch(carboxylic_acid_pattern):
                    has_carboxylic_acid = True
                    break
            
            return has_ethyl_ester and has_carboxylic_acid
            
        except:
            return False
    
    def detect_esterification(self, rxn):
        """Detect carboxylic acid esterification to methyl ester"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0].split(".")
        products = rxn_parts[1]
        
        # Check for carboxylic acid in reactants
        carboxylic_acid_pattern = Chem.MolFromSmarts("C(=O)O")
        methyl_ester_pattern = Chem.MolFromSmarts("C(=O)OC")
        
        try:
            has_carboxylic_acid = False
            for reactant_smiles in reactants:
                reactant_mol = Chem.MolFromSmiles(reactant_smiles)
                if reactant_mol and reactant_mol.HasSubstructMatch(carboxylic_acid_pattern):
                    has_carboxylic_acid = True
                    break
            
            # Check for methyl ester in products
            product_mol = Chem.MolFromSmiles(products)
            if not product_mol:
                return False
                
            has_methyl_ester = product_mol.HasSubstructMatch(methyl_ester_pattern)
            
            return has_carboxylic_acid and has_methyl_ester
            
        except:
            return False
