"""Generated evaluation code for: Early tribromopyrazine assembly via amine intermediates"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlyTribromopyrazineAssembly(MultiRxnCondBase):
    """
    Evaluates routes for early tribromopyrazine assembly via amine intermediates.
    Checks for the presence of Curtius rearrangement, Boc deprotection, and Sandmeyer 
    reaction sequence that converts carboxylic acid to amine then to bromide.
    """
    
    def __init__(self, config):
        self.require_curtius = config.get("require_curtius", True)
        self.require_boc_deprotection = config.get("require_boc_deprotection", True)
        self.require_sandmeyer = config.get("require_sandmeyer", True)
        self.target_timing = config.get("timing", "early")  # "early" means lower depth is better
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Check for each reaction type in the sequence
        has_curtius = any(self.detect_curtius_rearrangement(r) for r in reactions)
        has_boc_deprotection = any(self.detect_boc_deprotection(r) for r in reactions)
        has_sandmeyer = any(self.detect_sandmeyer(r) for r in reactions)
        
        # Check if tribromopyrazine pattern is present
        has_tribromopyrazine = any(self.detect_tribromopyrazine_formation(r) for r in reactions)
        
        # All required conditions must be met
        condition = (
            (has_curtius == self.require_curtius) and
            (has_boc_deprotection == self.require_boc_deprotection) and
            (has_sandmeyer == self.require_sandmeyer) and
            has_tribromopyrazine
        )
        
        return condition, len(reactions)
    
    def detect_curtius_rearrangement(self, rxn):
        """Detects Curtius rearrangement: COOH -> NH2 via acyl azide intermediate"""
        prod_mol, react_mols = self.parse_reaction(rxn)
        
        # Look for carboxylic acid in reactants and amine in products
        carboxylic_acid_pattern = Chem.MolFromSmarts("[C](=[O])[OH]")
        amine_pattern = Chem.MolFromSmarts("[NH2]")
        
        has_carboxylic_acid = any(mol.HasSubstructMatch(carboxylic_acid_pattern) for mol in react_mols)
        has_amine = prod_mol.HasSubstructMatch(amine_pattern)
        
        return has_carboxylic_acid and has_amine
    
    def detect_boc_deprotection(self, rxn):
        """Detects Boc deprotection: NHBoc -> NH2"""
        prod_mol, react_mols = self.parse_reaction(rxn)
        
        # Boc protecting group pattern
        boc_pattern = Chem.MolFromSmarts("[NH1][C](=[O])[O][C]([CH3])([CH3])[CH3]")
        free_amine_pattern = Chem.MolFromSmarts("[NH2]")
        
        has_boc = any(mol.HasSubstructMatch(boc_pattern) for mol in react_mols)
        has_free_amine = prod_mol.HasSubstructMatch(free_amine_pattern)
        
        return has_boc and has_free_amine
    
    def detect_sandmeyer(self, rxn):
        """Detects Sandmeyer reaction: NH2 -> Br via diazonium intermediate"""
        prod_mol, react_mols = self.parse_reaction(rxn)
        
        # Aniline pattern (aromatic amine) in reactants
        aniline_pattern = Chem.MolFromSmarts("[c][NH2]")
        # Aryl bromide in products
        aryl_bromide_pattern = Chem.MolFromSmarts("[c][Br]")
        
        has_aniline = any(mol.HasSubstructMatch(aniline_pattern) for mol in react_mols)
        has_aryl_bromide = prod_mol.HasSubstructMatch(aryl_bromide_pattern)
        
        return has_aniline and has_aryl_bromide
    
    def detect_tribromopyrazine_formation(self, rxn):
        """Detects formation of tribromopyrazine structure"""
        prod_mol, react_mols = self.parse_reaction(rxn)
        
        # Tribromopyrazine pattern: pyrazine ring with three bromines
        tribromopyrazine_pattern = Chem.MolFromSmarts("[c]1[c]([Br])[n][c]([Br])[c]([Br])[n]1")
        
        return prod_mol.HasSubstructMatch(tribromopyrazine_pattern)
    
    def parse_reaction(self, rxn):
        """Helper method to parse reaction SMILES into product and reactant molecules"""
        rxn_parts = rxn.split(">>")
        prod_mol = Chem.MolFromSmiles(rxn_parts[0])
        react_mols = [Chem.MolFromSmiles(r) for r in rxn_parts[1].split(".")]
        return prod_mol, react_mols
