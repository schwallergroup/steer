"""Generated evaluation code for: Orthogonal benzyl and tert-butyl ester protection strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class OrthogonalProtectingGroupStrategy(MultiRxnCondBase):
    """
    Evaluates whether a synthesis route employs an orthogonal protecting group strategy
    using both benzyl and tert-butyl esters. Checks for the presence of both protecting
    groups and their selective deprotection reactions.
    """
    
    def __init__(self, config):
        self.required_groups = config.get("protecting_groups", ["benzyl_ester", "tert_butyl_ester"])
        self.strategy = config.get("strategy", "orthogonal")
        
        # SMARTS patterns for protecting groups
        self.benzyl_ester_pattern = "[CH2]([cH1][cH1][cH1][cH1][cH1][cH1]1)OC(=O)"  # Benzyl ester
        self.tert_butyl_ester_pattern = "C(C)(C)(C)OC(=O)"  # tert-Butyl ester
        
        # SMARTS patterns for deprotection products
        self.carboxylic_acid_pattern = "C(=O)O"  # Carboxylic acid from deprotection
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Check for presence of both protecting groups in route
        has_benzyl = any(self.detect_benzyl_ester(r) for r in reactions)
        has_tert_butyl = any(self.detect_tert_butyl_ester(r) for r in reactions)
        
        # Check for selective deprotection reactions
        has_benzyl_deprotection = any(self.detect_benzyl_deprotection(r) for r in reactions)
        has_tert_butyl_deprotection = any(self.detect_tert_butyl_deprotection(r) for r in reactions)
        
        # For orthogonal strategy, we need both protecting groups present
        # and evidence of selective deprotection
        if self.strategy == "orthogonal":
            both_groups_present = has_benzyl and has_tert_butyl
            selective_deprotection = has_benzyl_deprotection or has_tert_butyl_deprotection
            condition = both_groups_present and selective_deprotection
        else:
            condition = has_benzyl or has_tert_butyl
        
        return condition, len(reactions)
    
    def detect_benzyl_ester(self, rxn):
        """Detect presence of benzyl ester protecting group in reaction"""
        return self.detect_pattern_in_reaction(rxn, self.benzyl_ester_pattern)
    
    def detect_tert_butyl_ester(self, rxn):
        """Detect presence of tert-butyl ester protecting group in reaction"""
        return self.detect_pattern_in_reaction(rxn, self.tert_butyl_ester_pattern)
    
    def detect_benzyl_deprotection(self, rxn):
        """Detect benzyl ester deprotection (benzyl ester -> carboxylic acid)"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        # Check if reactant has benzyl ester and product has carboxylic acid
        reactant_mol = Chem.MolFromSmiles(reactants)
        if reactant_mol is None:
            return False
            
        has_benzyl_in_reactant = reactant_mol.HasSubstructMatch(
            Chem.MolFromSmarts(self.benzyl_ester_pattern)
        )
        
        # Check products for carboxylic acid formation
        product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".")]
        has_acid_in_products = any(
            mol and mol.HasSubstructMatch(Chem.MolFromSmarts(self.carboxylic_acid_pattern))
            for mol in product_mols if mol
        )
        
        return has_benzyl_in_reactant and has_acid_in_products
    
    def detect_tert_butyl_deprotection(self, rxn):
        """Detect tert-butyl ester deprotection (tert-butyl ester -> carboxylic acid)"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        # Check if reactant has tert-butyl ester and product has carboxylic acid
        reactant_mol = Chem.MolFromSmiles(reactants)
        if reactant_mol is None:
            return False
            
        has_tert_butyl_in_reactant = reactant_mol.HasSubstructMatch(
            Chem.MolFromSmarts(self.tert_butyl_ester_pattern)
        )
        
        # Check products for carboxylic acid formation
        product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".")]
        has_acid_in_products = any(
            mol and mol.HasSubstructMatch(Chem.MolFromSmarts(self.carboxylic_acid_pattern))
            for mol in product_mols if mol
        )
        
        return has_tert_butyl_in_reactant and has_acid_in_products
    
    def detect_pattern_in_reaction(self, rxn, pattern_smarts):
        """Helper method to detect a SMARTS pattern anywhere in a reaction"""
        pattern = Chem.MolFromSmarts(pattern_smarts)
        if pattern is None:
            return False
            
        # Check all molecules in the reaction
        all_smiles = rxn.replace(">>", ".").split(".")
        for smi in all_smiles:
            mol = Chem.MolFromSmiles(smi.strip())
            if mol and mol.HasSubstructMatch(pattern):
                return True
        return False
