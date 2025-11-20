"""Generated evaluation code for: Early stage biphenyl formation via Suzuki"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlyStageBiphenylSuzuki(BaseScoring):
    """
    Evaluates whether biphenyl formation via Suzuki coupling occurs early in the synthesis.
    Returns higher scores when Suzuki coupling creating biphenyl bonds happens at earlier stages.
    """
    
    def __init__(self, config: Dict):
        self.target_stage = config.get("target_stage", "early")
        self.max_depth_fraction = config.get("max_depth_fraction", 0.3)  # Early = first 30% of route
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Suzuki biphenyl formation doesn't happen
        
        if self.target_stage == "early":
            if x <= self.max_depth_fraction:
                return 10  # Perfect score for early stage
            else:
                # Penalize later stage reactions
                return max(0, 10 * (1 - x))
        else:
            # For other stages, inverse scoring
            return 10 * x
    
    def hit_condition(self, d):
        """Check if this reaction is a Suzuki coupling forming a biphenyl bond"""
        metadata = d.get("metadata", {})
        rxn_smiles = metadata.get("mapped_reaction_smiles", "")
        
        if not rxn_smiles or ">>" not in rxn_smiles:
            return False
        
        try:
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse molecules
            product = Chem.MolFromSmiles(products_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check for Suzuki coupling indicators
            if not self._is_suzuki_coupling(reactants, product):
                return False
            
            # Check for biphenyl formation
            return self._forms_biphenyl_bond(reactants, product)
            
        except Exception:
            return False
    
    def _is_suzuki_coupling(self, reactants, product):
        """Check if reaction has Suzuki coupling characteristics"""
        # Look for boronic acid/ester and halide reactants
        boronic_pattern = Chem.MolFromSmarts("[#6][B]([OH])([OH])")  # Boronic acid
        boronic_ester_pattern = Chem.MolFromSmarts("[#6][B]1OC([CH3])([CH3])C([CH3])([CH3])O1")  # Pinacol ester
        halide_pattern = Chem.MolFromSmarts("[#6][Br,I,Cl]")  # Aryl halide
        
        has_boron = False
        has_halide = False
        
        for reactant in reactants:
            if reactant.HasSubstructMatch(boronic_pattern) or reactant.HasSubstructMatch(boronic_ester_pattern):
                has_boron = True
            if reactant.HasSubstructMatch(halide_pattern):
                has_halide = True
        
        return has_boron and has_halide
    
    def _forms_biphenyl_bond(self, reactants, product):
        """Check if a new biphenyl (biaryl) bond is formed"""
        # Biphenyl pattern - two aromatic rings connected by single bond
        biphenyl_pattern = Chem.MolFromSmarts("c1ccccc1-c2ccccc2")
        
        # Product should contain biphenyl
        if not product.HasSubstructMatch(biphenyl_pattern):
            return False
        
        # Check that biphenyl wasn't present in reactants (new bond formation)
        for reactant in reactants:
            if reactant.HasSubstructMatch(biphenyl_pattern):
                return False  # Biphenyl already existed
        
        # Additional check for aromatic carbon-carbon bond formation
        aromatic_c_pattern = Chem.MolFromSmarts("[c]")
        
        # Count aromatic carbons in reactants vs product
        reactant_ar_c = sum(len(r.GetSubstructMatches(aromatic_c_pattern)) for r in reactants)
        product_ar_c = len(product.GetSubstructMatches(aromatic_c_pattern))
        
        # In Suzuki coupling, we expect similar aromatic carbon counts (no new aromatics formed)
        # but new C-C bond between existing aromatic systems
        return abs(reactant_ar_c - product_ar_c) <= 2  # Allow small difference for leaving groups
