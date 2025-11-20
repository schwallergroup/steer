"""Generated evaluation code for: Methyl ester protecting group strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class MethylEsterProtectingStrategy(BaseScoring):
    """
    Evaluates synthesis routes for methyl ester protecting group strategy.
    Checks if carboxylic acid is protected as methyl ester and deprotected
    within the specified number of steps.
    """
    
    def __init__(self, config: Dict):
        self.deprotection_steps = config["parameters"]["deprotection_steps"]
        # SMARTS patterns for detection
        self.carboxylic_acid_pattern = "[CX3](=O)[OX2H1]"
        self.methyl_ester_pattern = "[CX3](=O)[OX2][CH3]"
        self.amide_pattern = "[CX3](=O)[NX3]"
        self.nitrile_pattern = "[CX2]#[NX1]"
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Strategy not found
        # Reward earlier implementation of protecting group strategy
        return max(0, 10 - x * 10)
    
    def hit_condition(self, d):
        """Check if this reaction implements methyl ester protecting group strategy"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            reactants_smiles, products_smiles = mapped_rxn.split(">>")
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            # Filter out None molecules
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            if not reactants or not products:
                return False
                
            # Check for protection: carboxylic acid -> methyl ester
            if self._is_protection_step(reactants, products):
                return self._validate_deprotection_pathway(d)
                
            return False
            
        except Exception:
            return False
    
    def _is_protection_step(self, reactants, products):
        """Check if reaction converts carboxylic acid to methyl ester"""
        carboxyl_in_reactants = any(
            mol.HasSubstructMatch(Chem.MolFromSmarts(self.carboxylic_acid_pattern))
            for mol in reactants
        )
        
        methyl_ester_in_products = any(
            mol.HasSubstructMatch(Chem.MolFromSmarts(self.methyl_ester_pattern))
            for mol in products
        )
        
        return carboxyl_in_reactants and methyl_ester_in_products
    
    def _validate_deprotection_pathway(self, current_node):
        """Validate that deprotection occurs within specified steps"""
        # Traverse forward in the synthesis tree to check for deprotection
        return self._check_deprotection_in_descendants(current_node, 0)
    
    def _check_deprotection_in_descendants(self, node, current_depth):
        """Recursively check descendants for deprotection pathway"""
        if current_depth >= self.deprotection_steps:
            return False
            
        # Check if current node shows deprotection
        if self._shows_deprotection_step(node):
            return True
            
        # Check children
        children = node.get("children", [])
        for child in children:
            if self._check_deprotection_in_descendants(child, current_depth + 1):
                return True
                
        return False
    
    def _shows_deprotection_step(self, node):
        """Check if node shows part of the deprotection pathway"""
        metadata = node.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            reactants_smiles, products_smiles = mapped_rxn.split(">>")
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            if not reactants or not products:
                return False
            
            # Check for methyl ester -> amide conversion
            methyl_ester_in_reactants = any(
                mol.HasSubstructMatch(Chem.MolFromSmarts(self.methyl_ester_pattern))
                for mol in reactants
            )
            amide_in_products = any(
                mol.HasSubstructMatch(Chem.MolFromSmarts(self.amide_pattern))
                for mol in products
            )
            
            # Check for amide -> nitrile conversion  
            amide_in_reactants = any(
                mol.HasSubstructMatch(Chem.MolFromSmarts(self.amide_pattern))
                for mol in reactants
            )
            nitrile_in_products = any(
                mol.HasSubstructMatch(Chem.MolFromSmarts(self.nitrile_pattern))
                for mol in products
            )
            
            # Check for nitrile -> carboxylic acid regeneration
            nitrile_in_reactants = any(
                mol.HasSubstructMatch(Chem.MolFromSmarts(self.nitrile_pattern))
                for mol in reactants
            )
            carboxyl_in_products = any(
                mol.HasSubstructMatch(Chem.MolFromSmarts(self.carboxylic_acid_pattern))
                for mol in products
            )
            
            return ((methyl_ester_in_reactants and amide_in_products) or
                    (amide_in_reactants and nitrile_in_products) or
                    (nitrile_in_reactants and carboxyl_in_products))
                    
        except Exception:
            return False
