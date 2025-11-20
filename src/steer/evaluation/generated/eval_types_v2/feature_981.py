"""Generated evaluation code for: Multi-step nitrile installation via amide intermediate"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class MultiStepNitrileInstallation(MultiRxnCondBase):
    """
    Checks for a multi-step nitrile installation via amide intermediate.
    Looks for a sequence: ester hydrolysis -> amidation -> dehydration to form nitrile.
    """
    
    def __init__(self, config):
        self.reaction_sequence = config.get("reaction_sequence", ["ester_hydrolysis", "amidation", "dehydration"])
        self.target_group = config.get("target_group", "nitrile")
        self.step_count = config.get("step_count", 4)
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Check if we have the expected sequence
        has_ester_hydrolysis = any(self.detect_ester_hydrolysis(r) for r in reactions)
        has_amidation = any(self.detect_amidation(r) for r in reactions)
        has_dehydration = any(self.detect_dehydration_to_nitrile(r) for r in reactions)
        has_nitrile_formation = any(self.detect_nitrile_formation(r) for r in reactions)
        
        # Check if we have at least 3 of the 4 key transformations
        sequence_steps = sum([has_ester_hydrolysis, has_amidation, has_dehydration, has_nitrile_formation])
        
        # Condition is met if we have the multi-step sequence (at least 3 steps)
        condition = sequence_steps >= 3 and len(reactions) >= self.step_count
        
        return condition, len(reactions)
    
    def detect_ester_hydrolysis(self, rxn):
        """Detect ester -> carboxylic acid transformation"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        # Look for ester pattern in reactants
        ester_pattern = Chem.MolFromSmarts("[C](=[O])[O][C]")
        # Look for carboxylic acid pattern in products
        acid_pattern = Chem.MolFromSmarts("[C](=[O])[OH]")
        
        try:
            react_mol = Chem.MolFromSmiles(reactants)
            prod_mol = Chem.MolFromSmiles(products.split(".")[0])  # Take first product
            
            if react_mol and prod_mol:
                has_ester = react_mol.HasSubstructMatch(ester_pattern)
                has_acid = prod_mol.HasSubstructMatch(acid_pattern)
                return has_ester and has_acid
        except:
            pass
        
        return False
    
    def detect_amidation(self, rxn):
        """Detect carboxylic acid -> amide transformation"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        # Look for carboxylic acid in reactants
        acid_pattern = Chem.MolFromSmarts("[C](=[O])[OH]")
        # Look for amide in products
        amide_pattern = Chem.MolFromSmarts("[C](=[O])[NH]")
        
        try:
            react_mol = Chem.MolFromSmiles(reactants.split(".")[0])  # Take first reactant
            prod_mol = Chem.MolFromSmiles(products)
            
            if react_mol and prod_mol:
                has_acid = react_mol.HasSubstructMatch(acid_pattern)
                has_amide = prod_mol.HasSubstructMatch(amide_pattern)
                return has_acid and has_amide
        except:
            pass
        
        return False
    
    def detect_dehydration_to_nitrile(self, rxn):
        """Detect amide -> nitrile dehydration"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        # Look for amide in reactants
        amide_pattern = Chem.MolFromSmarts("[C](=[O])[NH2]")  # Primary amide
        # Look for nitrile in products
        nitrile_pattern = Chem.MolFromSmarts("[C]#[N]")
        
        try:
            react_mol = Chem.MolFromSmiles(reactants.split(".")[0])  # Take first reactant
            prod_mol = Chem.MolFromSmiles(products.split(".")[0])   # Take first product
            
            if react_mol and prod_mol:
                has_amide = react_mol.HasSubstructMatch(amide_pattern)
                has_nitrile = prod_mol.HasSubstructMatch(nitrile_pattern)
                return has_amide and has_nitrile
        except:
            pass
        
        return False
    
    def detect_nitrile_formation(self, rxn):
        """Detect any transformation that forms a nitrile group"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        products = rxn_parts[1]
        nitrile_pattern = Chem.MolFromSmarts("[C]#[N]")
        
        try:
            prod_mols = [Chem.MolFromSmiles(p) for p in products.split(".")]
            return any(mol and mol.HasSubstructMatch(nitrile_pattern) for mol in prod_mols)
        except:
            pass
        
        return False
