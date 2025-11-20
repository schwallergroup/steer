"""Generated evaluation code for: Late stage Suzuki coupling convergent assembly"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageSuzukiConvergent(BaseScoring):
    """
    Evaluates if a Suzuki coupling occurs late in the synthesis route as a convergent assembly step.
    Checks for Suzuki coupling patterns and ensures it happens near the final steps.
    """
    
    def __init__(self, config: Dict):
        self.min_fragments = config.get("fragments", 2)
        self.stage_threshold = config.get("stage_threshold", 0.8)  # Late stage = top 20% of route
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # No Suzuki coupling found
        else:
            # Late stage coupling is better (x closer to 1.0)
            if x >= self.stage_threshold:
                return 1.0 - (1.0 - x) * 5  # Reward very late stage
            else:
                return x * 0.5  # Penalize early stage coupling
    
    def hit_condition(self, d):
        """Check if this reaction is a Suzuki coupling with convergent assembly"""
        metadata = d.get("metadata", {})
        rxn_smiles = metadata.get("mapped_reaction_smiles", "")
        
        if not rxn_smiles or ">>" not in rxn_smiles:
            return False
            
        try:
            prod_smiles, react_smiles = rxn_smiles.split(">>")
            reactants = [Chem.MolFromSmiles(r.strip()) for r in react_smiles.split(".")]
            product = Chem.MolFromSmiles(prod_smiles)
            
            if not product or len(reactants) < self.min_fragments:
                return False
            
            # Check for Suzuki coupling pattern
            if not self._is_suzuki_coupling(reactants, product):
                return False
                
            # Check for convergent assembly (both fragments should be substantial)
            return self._is_convergent_assembly(reactants)
            
        except Exception:
            return False
    
    def _is_suzuki_coupling(self, reactants, product):
        """Detect Suzuki coupling by looking for boronic acid/ester and halide patterns"""
        # Boronic acid/ester patterns
        boronic_patterns = [
            Chem.MolFromSmarts("[#6]-B(-O)(-O)"),  # Boronic ester
            Chem.MolFromSmarts("[#6]-B(-[OH])(-[OH])"),  # Boronic acid
            Chem.MolFromSmarts("[#6]-B1-O-C-C-O-1")  # Pinacol boronate
        ]
        
        # Halide patterns (typically Br, I, sometimes Cl with electron-withdrawing groups)
        halide_patterns = [
            Chem.MolFromSmarts("[#6]-Br"),
            Chem.MolFromSmarts("[#6]-I"),
            Chem.MolFromSmarts("c-Br"),  # Aromatic bromide
            Chem.MolFromSmarts("c-I")    # Aromatic iodide
        ]
        
        has_boronic = False
        has_halide = False
        
        for reactant in reactants:
            if reactant is None:
                continue
                
            # Check for boronic acid/ester
            for pattern in boronic_patterns:
                if reactant.HasSubstructMatch(pattern):
                    has_boronic = True
                    break
                    
            # Check for halide
            for pattern in halide_patterns:
                if reactant.HasSubstructMatch(pattern):
                    has_halide = True
                    break
                    
        return has_boronic and has_halide
    
    def _is_convergent_assembly(self, reactants):
        """Check if reactants represent substantial fragments (convergent vs linear)"""
        substantial_fragments = 0
        min_size = 6  # Minimum atoms to be considered a substantial fragment
        
        for reactant in reactants:
            if reactant is None:
                continue
                
            # Count heavy atoms (non-hydrogen)
            heavy_atoms = reactant.GetNumHeavyAtoms()
            
            # Skip small reagents/catalysts
            if heavy_atoms >= min_size:
                substantial_fragments += 1
                
        return substantial_fragments >= self.min_fragments
