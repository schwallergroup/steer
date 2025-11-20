"""Generated evaluation code for: Convergent synthesis via Suzuki coupling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentSuzukiCoupling(BaseScoring):
    """
    Evaluates convergent synthesis strategy using Suzuki coupling.
    Checks if a Suzuki-Miyaura cross-coupling reaction occurs at the specified timing
    to join two major fragments forming a biaryl core.
    """
    
    def __init__(self, config: Dict):
        self.target_fragments = config["parameters"].get("fragments", 2)
        self.timing = config["parameters"].get("timing", "middle")
        
        # Define timing thresholds
        if self.timing == "early":
            self.target_depth_range = (0.0, 0.3)
        elif self.timing == "middle":
            self.target_depth_range = (0.3, 0.7)
        else:  # late
            self.target_depth_range = (0.7, 1.0)

    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Suzuki coupling doesn't occur
        
        # Check if depth is within target timing range
        if self.target_depth_range[0] <= x <= self.target_depth_range[1]:
            return 1.0  # Perfect timing
        else:
            # Penalize based on distance from target range
            if x < self.target_depth_range[0]:
                distance = self.target_depth_range[0] - x
            else:
                distance = x - self.target_depth_range[1]
            return max(0, 1.0 - distance * 2)

    def hit_condition(self, d) -> bool:
        """Check if this reaction is a Suzuki coupling forming biaryl bond."""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            prod_smiles, react_smiles = mapped_rxn.split(">>")
            product = Chem.MolFromSmiles(prod_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in react_smiles.split(".")]
            
            if not product or len(reactants) < 2:
                return False
                
            # Check for Suzuki coupling characteristics
            if not self._is_suzuki_coupling(reactants):
                return False
                
            # Check if reaction joins expected number of fragments
            major_fragments = self._count_major_fragments(reactants)
            if major_fragments != self.target_fragments:
                return False
                
            # Check if biaryl bond is formed
            return self._forms_biaryl_bond(product, reactants)
            
        except Exception:
            return False

    def _is_suzuki_coupling(self, reactants) -> bool:
        """Check if reactants contain Suzuki coupling partners."""
        # Boronic acid/ester pattern
        boronic_pattern = Chem.MolFromSmarts("[#6]-B(-O)(-O)")
        boronate_pattern = Chem.MolFromSmarts("[#6]-B1-O-C-C-O-1")
        
        # Aryl halide pattern
        aryl_halide_pattern = Chem.MolFromSmarts("c[F,Cl,Br,I]")
        
        has_boron = False
        has_halide = False
        
        for reactant in reactants:
            if reactant.HasSubstructMatch(boronic_pattern) or reactant.HasSubstructMatch(boronate_pattern):
                has_boron = True
            elif reactant.HasSubstructMatch(aryl_halide_pattern):
                has_halide = True
                
        return has_boron and has_halide

    def _count_major_fragments(self, reactants) -> int:
        """Count major organic fragments (excluding small molecules/catalysts)."""
        major_fragments = 0
        
        for reactant in reactants:
            # Skip small molecules typically used in Suzuki coupling
            if reactant.GetNumAtoms() >= 6:  # Arbitrary cutoff for "major" fragments
                # Check if it's not a typical catalyst/base
                if not self._is_catalyst_or_base(reactant):
                    major_fragments += 1
                    
        return major_fragments

    def _is_catalyst_or_base(self, mol) -> bool:
        """Check if molecule is likely a catalyst or base."""
        # Common Suzuki bases and catalysts
        base_patterns = [
            "C(C)(C)(C)[O-]",  # tert-butoxide
            "[Na+]",  # sodium salts
            "[K+]",   # potassium salts
            "[Cs+]",  # cesium salts
        ]
        
        catalyst_patterns = [
            "[Pd]",  # Palladium catalysts
        ]
        
        smiles = Chem.MolToSmiles(mol)
        
        for pattern in base_patterns + catalyst_patterns:
            try:
                if Chem.MolFromSmarts(pattern) and mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                    return True
            except:
                continue
                
        return False

    def _forms_biaryl_bond(self, product, reactants) -> bool:
        """Check if a new biaryl bond is formed in the product."""
        # Biaryl pattern (aromatic C-C bond between rings)
        biaryl_pattern = Chem.MolFromSmarts("c-c")
        
        if not product.HasSubstructMatch(biaryl_pattern):
            return False
            
        # Check that this biaryl bond is new (not present in individual reactants)
        product_biaryl_count = len(product.GetSubstructMatches(biaryl_pattern))
        
        reactant_biaryl_total = 0
        for reactant in reactants:
            if self._is_catalyst_or_base(reactant):
                continue
            reactant_biaryl_total += len(reactant.GetSubstructMatches(biaryl_pattern))
            
        # New biaryl bond formed if product has more than sum of reactants
        return product_biaryl_count > reactant_biaryl_total
