"""Generated evaluation code for: Convergent synthesis via two fragments"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentStrategy(BaseScoring):
    """
    Evaluates convergent synthesis strategy by checking if two complex fragments
    are coupled together in a late-stage reaction.
    
    A convergent reaction is identified when:
    1. Two or more reactants each have complexity >= min_fragment_complexity
    2. The reaction occurs at depth <= coupling_stage_threshold (late-stage)
    3. The reactants are combined to form a more complex product
    """
    
    def __init__(self, config: Dict):
        self.min_fragment_complexity = config.get("min_fragment_complexity", 5)
        self.coupling_stage_threshold = config.get("coupling_stage_threshold", 0.7)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # No convergent coupling found
        else:
            # Earlier convergent coupling (lower depth) is better
            # Scale to 0-10 range, with early coupling scoring higher
            return max(0, 10 * (1 - x))
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction represents a convergent coupling"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            product_smiles = rxn_parts[0]
            reactants_smiles = rxn_parts[1].split(".")
            
            # Need at least 2 reactants for convergent synthesis
            if len(reactants_smiles) < 2:
                return False
                
            # Calculate complexity for each reactant
            complex_fragments = 0
            for reactant_smiles in reactants_smiles:
                mol = Chem.MolFromSmiles(reactant_smiles)
                if mol is not None:
                    complexity = self._calculate_complexity(mol)
                    if complexity >= self.min_fragment_complexity:
                        complex_fragments += 1
            
            # Check if we have at least 2 complex fragments being coupled
            return complex_fragments >= 2
            
        except Exception:
            return False
    
    def _calculate_complexity(self, mol) -> int:
        """
        Calculate molecular complexity based on:
        - Number of rings
        - Number of heteroatoms
        - Number of rotatable bonds
        - Molecular weight contribution
        """
        if mol is None:
            return 0
            
        ring_info = mol.GetRingInfo()
        num_rings = ring_info.NumRings()
        
        # Count heteroatoms (non C,H)
        heteroatoms = sum(1 for atom in mol.GetAtoms() 
                         if atom.GetAtomicNum() not in [1, 6])
        
        # Count rotatable bonds
        rotatable_bonds = Descriptors.NumRotatableBonds(mol)
        
        # Heavy atom count as size factor
        heavy_atoms = mol.GetNumHeavyAtoms()
        
        # Weighted complexity score
        complexity = (num_rings * 2 + 
                     heteroatoms * 1.5 + 
                     rotatable_bonds * 0.5 + 
                     heavy_atoms * 0.1)
        
        return int(complexity)
