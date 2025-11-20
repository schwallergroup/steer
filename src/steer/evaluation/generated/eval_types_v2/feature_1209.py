"""Generated evaluation code for: Ester hydrolysis re-esterification sequence"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EsterHydrolysisReesterification(MultiRxnCondBase):
    """
    Detects ester hydrolysis followed by re-esterification sequence in synthesis routes.
    Specifically looks for methyl ester hydrolysis followed by ester reformation.
    """
    
    def __init__(self, config):
        self.substrate = config.get("substrate", "methyl_ester")
        self.allow_sequence = config.get("allow_sequence", True)
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Find hydrolysis and esterification reactions
        hydrolysis_indices = []
        esterification_indices = []
        
        for i, rxn in enumerate(reactions):
            if self.detect_ester_hydrolysis(rxn):
                hydrolysis_indices.append(i)
            if self.detect_esterification(rxn):
                esterification_indices.append(i)
        
        # Check if hydrolysis is followed by esterification
        sequence_found = False
        for hydro_idx in hydrolysis_indices:
            for ester_idx in esterification_indices:
                if ester_idx > hydro_idx:  # Esterification after hydrolysis
                    sequence_found = True
                    break
            if sequence_found:
                break
        
        condition = sequence_found == self.allow_sequence
        return condition, len(reactions)
    
    def detect_ester_hydrolysis(self, rxn):
        """Detect ester hydrolysis reaction (ester -> carboxylic acid)"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0].split(".")
        products = rxn_parts[1].split(".")
        
        # Look for methyl ester in reactants
        methyl_ester_pattern = Chem.MolFromSmarts("[#6][C](=O)[O][CH3]")
        carboxylic_acid_pattern = Chem.MolFromSmarts("[#6][C](=O)[OH]")
        
        has_methyl_ester_reactant = False
        has_carboxylic_acid_product = False
        
        for reactant_smi in reactants:
            try:
                mol = Chem.MolFromSmiles(reactant_smi)
                if mol and mol.HasSubstructMatch(methyl_ester_pattern):
                    has_methyl_ester_reactant = True
                    break
            except:
                continue
        
        for product_smi in products:
            try:
                mol = Chem.MolFromSmiles(product_smi)
                if mol and mol.HasSubstructMatch(carboxylic_acid_pattern):
                    has_carboxylic_acid_product = True
                    break
            except:
                continue
        
        return has_methyl_ester_reactant and has_carboxylic_acid_product
    
    def detect_esterification(self, rxn):
        """Detect esterification reaction (carboxylic acid -> ester)"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0].split(".")
        products = rxn_parts[1].split(".")
        
        # Look for carboxylic acid in reactants and ester in products
        carboxylic_acid_pattern = Chem.MolFromSmarts("[#6][C](=O)[OH]")
        ester_pattern = Chem.MolFromSmarts("[#6][C](=O)[O][#6]")
        
        has_carboxylic_acid_reactant = False
        has_ester_product = False
        
        for reactant_smi in reactants:
            try:
                mol = Chem.MolFromSmiles(reactant_smi)
                if mol and mol.HasSubstructMatch(carboxylic_acid_pattern):
                    has_carboxylic_acid_reactant = True
                    break
            except:
                continue
        
        for product_smi in products:
            try:
                mol = Chem.MolFromSmiles(product_smi)
                if mol and mol.HasSubstructMatch(ester_pattern):
                    has_ester_product = True
                    break
            except:
                continue
        
        return has_carboxylic_acid_reactant and has_ester_product
