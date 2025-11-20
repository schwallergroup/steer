"""Generated evaluation code for: Convergent biaryl fragment assembly"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentBiarylAssembly(BaseScoring):
    """
    Evaluates synthesis routes for convergent biaryl fragment assembly through cross-coupling reactions.
    Checks if two pre-formed aromatic fragments are joined via cross-coupling at an appropriate depth.
    """
    
    def __init__(self, config: Dict):
        self.convergence_point = config.get("convergence_point", "middle")
        self.ideal_depth_fraction = 0.5 if self.convergence_point == "middle" else 0.3
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Convergent biaryl assembly doesn't happen
        
        # Score based on how close to ideal convergence point
        deviation = abs(x - self.ideal_depth_fraction)
        return max(0, 1 - 2 * deviation)  # Scale to 0-1 range
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction represents convergent biaryl cross-coupling"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        rxn_parts = mapped_rxn.split(">>")
        product_smiles = rxn_parts[0]
        reactant_smiles = rxn_parts[1]
        
        try:
            product_mol = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactant_smiles.split(".")]
            
            if not product_mol or len(reactants) < 2:
                return False
                
            # Check if product contains biaryl motif
            biaryl_pattern = Chem.MolFromSmarts("c1ccccc1-c2ccccc2")
            if not product_mol.HasSubstructMatch(biaryl_pattern):
                return False
                
            # Check if this is a cross-coupling reaction (C-C bond formation between aromatics)
            if not self._is_cross_coupling_reaction(product_mol, reactants):
                return False
                
            # Check convergence: both reactants should be substantial fragments
            return self._check_convergent_assembly(reactants)
            
        except Exception:
            return False
    
    def _is_cross_coupling_reaction(self, product_mol, reactants) -> bool:
        """Check if reaction involves aromatic C-C bond formation typical of cross-coupling"""
        # Look for common cross-coupling patterns
        cross_coupling_patterns = [
            "[c:1][c:2]",  # General aromatic C-C bond
            "[c:1]-[c:2]", # Aromatic-aromatic bond
        ]
        
        for pattern_smarts in cross_coupling_patterns:
            pattern = Chem.MolFromSmarts(pattern_smarts)
            if product_mol.HasSubstructMatch(pattern):
                # Check if the bond being formed connects two aromatic systems
                matches = product_mol.GetSubstructMatches(pattern)
                for match in matches:
                    atom1, atom2 = match[0], match[1]
                    if self._atoms_in_different_aromatic_rings(product_mol, atom1, atom2):
                        return True
        return False
    
    def _atoms_in_different_aromatic_rings(self, mol, atom1_idx, atom2_idx) -> bool:
        """Check if two atoms are in different aromatic ring systems"""
        atom1 = mol.GetAtomWithIdx(atom1_idx)
        atom2 = mol.GetAtomWithIdx(atom2_idx)
        
        if not (atom1.GetIsAromatic() and atom2.GetIsAromatic()):
            return False
            
        # Get ring info
        ring_info = mol.GetRingInfo()
        atom1_rings = [ring for ring in ring_info.AtomRings() if atom1_idx in ring]
        atom2_rings = [ring for ring in ring_info.AtomRings() if atom2_idx in ring]
        
        # Check if atoms are in different ring systems
        return not any(ring1 == ring2 for ring1 in atom1_rings for ring2 in atom2_rings)
    
    def _check_convergent_assembly(self, reactants) -> bool:
        """Check if reactants represent convergent fragments (both should be substantial)"""
        if len(reactants) < 2:
            return False
            
        aromatic_reactants = []
        for reactant in reactants:
            if reactant and self._has_substantial_aromatic_system(reactant):
                aromatic_reactants.append(reactant)
        
        # For convergent assembly, need at least 2 substantial aromatic fragments
        return len(aromatic_reactants) >= 2
    
    def _has_substantial_aromatic_system(self, mol) -> bool:
        """Check if molecule has substantial aromatic system (not just simple aromatics)"""
        if not mol:
            return False
            
        aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
        heavy_atoms = mol.GetNumHeavyAtoms()
        
        # Substantial if has aromatic system and reasonable size
        return aromatic_atoms >= 6 and heavy_atoms >= 8
