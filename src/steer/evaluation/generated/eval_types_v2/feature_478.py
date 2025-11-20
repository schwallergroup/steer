"""Generated evaluation code for: Early stage Sonogashira coupling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlySonogashira(BaseScoring):
    """
    Evaluates whether a Sonogashira coupling reaction occurs in the early stages 
    of a synthesis route (within first 3 steps from target).
    """
    
    def __init__(self, config: Dict):
        self.max_depth = config.get("depth_range", [0, 3])[1] / 10.0  # Convert to fraction
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Sonogashira coupling doesn't occur
        elif x <= self.max_depth:
            return 10  # Perfect - occurs in early stage
        else:
            # Penalize later occurrence, score decreases as depth increases
            penalty = (x - self.max_depth) * 20  # Scale penalty
            return max(0, 10 - penalty)
    
    def hit_condition(self, d) -> bool:
        """
        Detect Sonogashira coupling by looking for:
        1. Formation of aryl-alkyne bond (C#C-aryl pattern)
        2. Presence of aryl halide and terminal alkyne as reactants
        """
        metadata = d.get("metadata", {})
        rxn_smiles = metadata.get("mapped_reaction_smiles", "")
        
        if not rxn_smiles or ">>" not in rxn_smiles:
            return False
            
        try:
            prod_smiles, react_smiles = rxn_smiles.split(">>")
            product = Chem.MolFromSmiles(prod_smiles)
            reactants = [Chem.MolFromSmiles(r) for r in react_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check if product contains aryl-alkyne bond
            aryl_alkyne_pattern = Chem.MolFromSmarts("c-C#C")
            if not product.HasSubstructMatch(aryl_alkyne_pattern):
                return False
            
            # Check reactants for typical Sonogashira components
            has_aryl_halide = False
            has_terminal_alkyne = False
            
            for reactant in reactants:
                # Check for aryl halide (Br, I, Cl on aromatic carbon)
                aryl_halide_patterns = [
                    Chem.MolFromSmarts("c-Br"),
                    Chem.MolFromSmarts("c-I"), 
                    Chem.MolFromSmarts("c-Cl")
                ]
                if any(reactant.HasSubstructMatch(pattern) for pattern in aryl_halide_patterns):
                    has_aryl_halide = True
                
                # Check for terminal alkyne
                terminal_alkyne_pattern = Chem.MolFromSmarts("C#C[H]")
                if reactant.HasSubstructMatch(terminal_alkyne_pattern):
                    has_terminal_alkyne = True
            
            return has_aryl_halide and has_terminal_alkyne
            
        except Exception:
            return False
