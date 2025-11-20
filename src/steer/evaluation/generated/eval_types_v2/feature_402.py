"""Generated evaluation code for: Convergent synthesis via two complex fragments"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentSynthesis(BaseScoring):
    """
    Evaluates convergent synthesis strategy where two complex fragments are built 
    separately and then coupled at a specific step. Checks if the convergence occurs
    at the target step and if both fragments meet complexity threshold.
    """
    
    def __init__(self, config: Dict):
        self.convergence_step = config["convergence_step"]
        self.fragment_complexity_threshold = config["fragment_complexity_threshold"]
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Convergent coupling doesn't happen
        else:
            # Perfect score if at target step, penalty for deviation
            step_penalty = abs(x - (self.convergence_step / 10.0))
            return max(0, 1.0 - step_penalty * 2)
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction represents a convergent coupling of two complex fragments"""
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            rxn_parts = rxn_smiles.split(">>")
            
            if len(rxn_parts) != 2:
                return False
                
            reactants = rxn_parts[1].split(".")
            
            # Need exactly 2 reactants for convergent coupling
            if len(reactants) != 2:
                return False
            
            # Check if both reactants meet complexity threshold
            complex_fragments = 0
            for reactant_smiles in reactants:
                reactant_mol = Chem.MolFromSmiles(reactant_smiles)
                if reactant_mol and self._is_complex_fragment(reactant_mol):
                    complex_fragments += 1
            
            # Both fragments must be complex for convergent synthesis
            return complex_fragments == 2
            
        except Exception:
            return False
    
    def _is_complex_fragment(self, mol) -> bool:
        """Determine if a molecule is a complex fragment based on various criteria"""
        if mol is None:
            return False
        
        complexity_score = 0
        
        # Count heavy atoms
        heavy_atom_count = mol.GetNumHeavyAtoms()
        if heavy_atom_count >= 8:
            complexity_score += 1
        
        # Count rings
        ring_info = mol.GetRingInfo()
        num_rings = ring_info.NumRings()
        if num_rings >= 1:
            complexity_score += 1
        
        # Count heteroatoms
        heteroatom_count = sum(1 for atom in mol.GetAtoms() 
                              if atom.GetAtomicNum() not in [1, 6])
        if heteroatom_count >= 2:
            complexity_score += 1
        
        # Count aromatic rings
        aromatic_rings = sum(1 for ring in ring_info.AtomRings() 
                           if all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring))
        if aromatic_rings >= 1:
            complexity_score += 1
        
        # Check for functional groups (carbonyls, esters, amides, etc.)
        carbonyl_pattern = Chem.MolFromSmarts("[#6]=[#8]")
        ester_pattern = Chem.MolFromSmarts("[#6](=[#8])[#8][#6]")
        amide_pattern = Chem.MolFromSmarts("[#6](=[#8])[#7]")
        
        if (mol.HasSubstructMatch(carbonyl_pattern) or 
            mol.HasSubstructMatch(ester_pattern) or 
            mol.HasSubstructMatch(amide_pattern)):
            complexity_score += 1
        
        return complexity_score >= self.fragment_complexity_threshold
