"""Generated evaluation code for: Mid-stage nitrene cyclization for core formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class NitreneCyclization(BaseScoring):
    """
    Evaluates routes for mid-stage nitrene cyclization that forms core ring structures.
    Detects nitrene insertion reactions (typically from azides or other nitrogen precursors)
    that create new rings through intramolecular C-H insertion.
    """
    
    def __init__(self, config: Dict):
        self.timing = config.get("timing", "mid")
        self.target_depth_range = (0.3, 0.7) if self.timing == "mid" else (0.0, 0.3)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # No nitrene cyclization found
        
        # Check if timing is appropriate (mid-stage preferred)
        if self.timing == "mid":
            if self.target_depth_range[0] <= x <= self.target_depth_range[1]:
                return 1.0  # Perfect mid-stage timing
            else:
                # Penalize early or late timing
                distance = min(abs(x - self.target_depth_range[0]), 
                             abs(x - self.target_depth_range[1]))
                return max(0, 1.0 - distance * 2)
        else:
            return 1.0 - x  # Earlier is better for non-mid timing
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction involves nitrene cyclization with ring formation."""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            reactants_smiles, products_smiles = mapped_rxn.split(">>")
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            
            if not all(reactants) or not all(products):
                return False
            
            # Check for nitrene precursors in reactants and ring formation
            has_nitrene_precursor = self._detect_nitrene_precursor(reactants)
            has_ring_formation = self._detect_ring_formation(reactants, products)
            has_core_formation = self._detect_core_formation(reactants, products)
            
            return has_nitrene_precursor and has_ring_formation and has_core_formation
            
        except Exception:
            return False
    
    def _detect_nitrene_precursor(self, reactants) -> bool:
        """Detect common nitrene precursors like azides, hydroxylamine derivatives, etc."""
        nitrene_patterns = [
            "[N-]=[N+]=[N-]",  # Azide
            "[NH2+][O-]",      # Hydroxylamine derivative
            "N(=O)=O",         # Nitro group (can form nitrene under reduction)
            "NS(=O)=O",        # Sulfonamide (photochemical nitrene source)
            "[N-]S(=O)(=O)",   # Sulfonyl azide
        ]
        
        for reactant in reactants:
            if reactant is None:
                continue
            for pattern in nitrene_patterns:
                if reactant.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                    return True
        return False
    
    def _detect_ring_formation(self, reactants, products) -> bool:
        """Check if new rings are formed in the reaction."""
        reactant_rings = sum(mol.GetRingInfo().NumRings() for mol in reactants if mol)
        product_rings = sum(mol.GetRingInfo().NumRings() for mol in products if mol)
        return product_rings > reactant_rings
    
    def _detect_core_formation(self, reactants, products) -> bool:
        """Check if a core tricyclic or polycyclic structure is formed."""
        # Look for formation of fused ring systems typical of carbazole-like cores
        core_patterns = [
            "c1ccc2c(c1)c3ccccc3n2",  # Carbazole core
            "c1ccc2c(c1)nc3ccccc23",  # Quinoline-like fused system
            "c1ccc2c(c1)c3ncccc3n2",  # Phenanthroline-like core
            "N1c2ccccc2c3ccccc13",    # General tricyclic nitrogen core
        ]
        
        # Check if products contain core structures that reactants don't
        for product in products:
            if product is None:
                continue
            for pattern in core_patterns:
                if product.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                    # Verify this core wasn't already present in reactants
                    core_in_reactants = any(
                        reactant and reactant.HasSubstructMatch(Chem.MolFromSmarts(pattern))
                        for reactant in reactants
                    )
                    if not core_in_reactants:
                        return True
        return False
