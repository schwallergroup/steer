"""Generated evaluation code for: Acetate protection before acid-catalyzed cyclization"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class AcetateProtectionBeforeCyclization(MultiRxnCondBase):
    """
    Checks if acetate protection of tertiary alcohol occurs before acid-catalyzed cyclization.
    Evaluates the timing and sequence of protecting group installation relative to cyclization.
    """
    
    def __init__(self, config):
        self.require_sequence = config.get("require_sequence", True)
        self.penalty_wrong_order = config.get("penalty_wrong_order", 0.5)
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Find acetate protection and cyclization reactions
        acetate_protection_indices = []
        cyclization_indices = []
        
        for i, rxn in enumerate(reactions):
            if self.detect_acetate_protection(rxn):
                acetate_protection_indices.append(i)
            if self.detect_acid_cyclization(rxn):
                cyclization_indices.append(i)
        
        # Check if both reactions are present
        has_protection = len(acetate_protection_indices) > 0
        has_cyclization = len(cyclization_indices) > 0
        
        if not (has_protection and has_cyclization):
            return False, len(reactions)
        
        # Check timing - protection should occur before cyclization
        # (higher index = earlier in synthesis)
        protection_depth = max(acetate_protection_indices)
        cyclization_depth = max(cyclization_indices)
        
        correct_timing = protection_depth > cyclization_depth
        
        if self.require_sequence and not correct_timing:
            return False, len(reactions)
        
        return True, len(reactions)
    
    def detect_acetate_protection(self, rxn):
        """Detect acetate protection of tertiary alcohol"""
        prod_mol = Chem.MolFromSmiles(rxn[0])
        react_mols = [Chem.MolFromSmiles(r) for r in rxn[1].split(".")]
        
        # Pattern for tertiary alcohol
        tertiary_alcohol_pattern = Chem.MolFromSmarts("[C](C)(C)(C)[OH]")
        
        # Pattern for acetate ester
        acetate_pattern = Chem.MolFromSmarts("[C](C)(C)(C)OC(=O)C")
        
        # Check if reactant has tertiary alcohol and product has acetate
        has_tertiary_alcohol = any(mol.HasSubstructMatch(tertiary_alcohol_pattern) 
                                 for mol in react_mols if mol is not None)
        has_acetate = prod_mol.HasSubstructMatch(acetate_pattern) if prod_mol else False
        
        # Additional check for acetylating reagent (Ac2O, AcCl, etc.)
        acetylating_reagents = [
            Chem.MolFromSmarts("CC(=O)OC(=O)C"),  # Ac2O
            Chem.MolFromSmarts("CC(=O)Cl"),       # AcCl
            Chem.MolFromSmarts("CC(=O)O")         # AcOH with coupling agent
        ]
        
        has_acetylating_agent = any(
            any(mol.HasSubstructMatch(pattern) for mol in react_mols if mol is not None)
            for pattern in acetylating_reagents
        )
        
        return has_tertiary_alcohol and has_acetate and has_acetylating_agent
    
    def detect_acid_cyclization(self, rxn):
        """Detect acid-catalyzed cyclization forming acetal/ketal"""
        prod_mol = Chem.MolFromSmiles(rxn[0])
        react_mols = [Chem.MolFromSmiles(r) for r in rxn[1].split(".")]
        
        # Pattern for acetal/ketal formation
        acetal_patterns = [
            Chem.MolFromSmarts("[C]1[O][C][O]1"),      # 5-membered acetal
            Chem.MolFromSmarts("[C]1[O][C][C][O]1"),   # 6-membered acetal
            Chem.MolFromSmarts("[C]1[O][C]([C])[O]1"), # ketal
        ]
        
        # Check if product contains cyclic acetal/ketal
        has_cyclic_acetal = any(prod_mol.HasSubstructMatch(pattern) 
                               for pattern in acetal_patterns if prod_mol)
        
        # Check for acid catalyst presence
        acid_catalysts = [
            Chem.MolFromSmarts("[H+]"),           # Generic acid
            Chem.MolFromSmarts("S(=O)(=O)(O)"),   # Sulfonic acid
            Chem.MolFromSmarts("P(=O)(O)(O)"),    # Phosphoric acid
            Chem.MolFromSmarts("[H]Cl"),          # HCl
        ]
        
        has_acid_catalyst = any(
            any(mol.HasSubstructMatch(pattern) for mol in react_mols if mol is not None)
            for pattern in acid_catalysts
        )
        
        # Check if reactant lacks the cyclic acetal (cyclization occurs)
        reactant_has_acetal = any(
            any(mol.HasSubstructMatch(pattern) for pattern in acetal_patterns)
            for mol in react_mols if mol is not None
        )
        
        return has_cyclic_acetal and has_acid_catalyst and not reactant_has_acetal
    
    def route_scoring(self, x):
        """Score based on presence and timing of protection strategy"""
        if x < 0:
            return 0  # Strategy not implemented
        else:
            return 1 - (x * self.penalty_wrong_order if not self.require_sequence else 0)
