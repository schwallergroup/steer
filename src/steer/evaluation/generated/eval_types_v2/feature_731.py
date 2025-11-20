"""Generated evaluation code for: Early stage [3+2] dipolar cycloaddition"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlyDipolarCycloaddition(BaseScoring):
    """
    Evaluates whether a [3+2] dipolar cycloaddition occurs early in the synthesis route.
    Returns higher scores when the cycloaddition happens at or before the specified depth threshold.
    """
    
    def __init__(self, config: Dict):
        self.depth_threshold = config["parameters"]["depth_threshold"]
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Cycloaddition doesn't happen
        
        # Convert depth to 0-1 scale where early reactions score higher
        depth_fraction = x
        
        # If cycloaddition occurs at or before threshold, give high score
        if depth_fraction <= (self.depth_threshold / 10.0):  # Normalize threshold
            return 10 * (1 - depth_fraction)  # Earlier is better
        else:
            # Penalty for late-stage cycloaddition
            return max(0, 10 * (1 - depth_fraction) * 0.3)
    
    def hit_condition(self, d) -> bool:
        """
        Detects [3+2] dipolar cycloaddition by checking for formation of 5-membered rings
        containing nitrogen and looking for characteristic reaction patterns.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
        
        try:
            reactants_smiles, product_smiles = mapped_rxn.split(">>")
            
            # Parse molecules
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check if product has more 5-membered rings than reactants
            product_5rings = self._count_5membered_rings(product)
            reactant_5rings = sum(self._count_5membered_rings(r) for r in reactants)
            
            if product_5rings <= reactant_5rings:
                return False
            
            # Look for nitrogen-containing 5-membered rings formed in product
            if not self._has_nitrogen_5ring(product):
                return False
            
            # Check for dipolar cycloaddition patterns
            return self._is_dipolar_cycloaddition_pattern(reactants, product)
            
        except Exception:
            return False
    
    def _count_5membered_rings(self, mol) -> int:
        """Count 5-membered rings in molecule."""
        if not mol:
            return 0
        
        ring_info = mol.GetRingInfo()
        return len([ring for ring in ring_info.AtomRings() if len(ring) == 5])
    
    def _has_nitrogen_5ring(self, mol) -> bool:
        """Check if molecule has nitrogen-containing 5-membered ring."""
        if not mol:
            return False
        
        ring_info = mol.GetRingInfo()
        for ring in ring_info.AtomRings():
            if len(ring) == 5:
                atoms_in_ring = [mol.GetAtomWithIdx(idx) for idx in ring]
                if any(atom.GetSymbol() == 'N' for atom in atoms_in_ring):
                    return True
        return False
    
    def _is_dipolar_cycloaddition_pattern(self, reactants, product) -> bool:
        """
        Check for characteristic [3+2] dipolar cycloaddition patterns:
        - Azomethine ylide + alkene
        - Nitrone + alkene
        - Azide + alkyne/alkene
        """
        if len(reactants) < 2:
            return False
        
        # Common dipolar patterns
        azomethine_ylide = Chem.MolFromSmarts("[N+]([C-])[C]")
        nitrone = Chem.MolFromSmarts("[N+]([O-])=C")
        azide = Chem.MolFromSmarts("[N-]=[N+]=[N-]")
        alkene = Chem.MolFromSmarts("C=C")
        alkyne = Chem.MolFromSmarts("C#C")
        
        dipolar_patterns = [azomethine_ylide, nitrone, azide]
        dipolarophile_patterns = [alkene, alkyne]
        
        has_dipole = False
        has_dipolarophile = False
        
        for reactant in reactants:
            # Check for 1,3-dipole
            if any(reactant.HasSubstructMatch(pattern) for pattern in dipolar_patterns if pattern):
                has_dipole = True
            
            # Check for dipolarophile
            if any(reactant.HasSubstructMatch(pattern) for pattern in dipolarophile_patterns if pattern):
                has_dipolarophile = True
        
        return has_dipole and has_dipolarophile
