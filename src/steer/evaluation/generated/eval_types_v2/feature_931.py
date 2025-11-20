"""Generated evaluation code for: Early Diels-Alder bicyclic core construction"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class DielsAlderEarlyCore(BaseScoring):
    """
    Evaluates if a Diels-Alder reaction occurs early in the synthesis route.
    
    Detects [4+2] cycloaddition reactions that form bicyclic structures and
    rewards earlier occurrence in the synthesis sequence.
    """
    
    def __init__(self, config: Dict):
        self.timing = config.get("timing", "early")
        self.target_depth_fraction = 0.2 if self.timing == "early" else 0.8
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # No Diels-Alder reaction found
        
        if self.timing == "early":
            # Reward early occurrence - lower depth fraction is better
            if x <= self.target_depth_fraction:
                return 10  # Perfect score for very early
            else:
                # Linear penalty as depth increases
                return max(0, 10 - 50 * (x - self.target_depth_fraction))
        else:
            # For late timing, higher depth fraction is better
            if x >= self.target_depth_fraction:
                return 10
            else:
                return max(0, 10 - 50 * (self.target_depth_fraction - x))
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction is a Diels-Alder cycloaddition."""
        metadata = d.get("metadata", {})
        rxn_smiles = metadata.get("mapped_reaction_smiles", "")
        
        if not rxn_smiles or ">>" not in rxn_smiles:
            return False
        
        try:
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            
            if not all(reactants) or not all(products):
                return False
            
            # Check if this is a Diels-Alder reaction
            return self._is_diels_alder_reaction(reactants, products)
            
        except Exception:
            return False
    
    def _is_diels_alder_reaction(self, reactants, products) -> bool:
        """
        Detect Diels-Alder reactions by checking for:
        1. Formation of 6-membered rings
        2. Bicyclic structure formation
        3. Pattern consistent with [4+2] cycloaddition
        """
        # Check if we have exactly 2 reactants forming 1 main product
        if len(reactants) != 2:
            return False
        
        # Count rings before and after reaction
        reactant_rings = sum(len(Chem.GetSymmSSSR(mol)) for mol in reactants)
        product_rings = sum(len(Chem.GetSymmSSSR(mol)) for mol in products)
        
        # Diels-Alder should increase ring count (typically by 1)
        if product_rings <= reactant_rings:
            return False
        
        # Look for bicyclic patterns in products
        bicyclic_patterns = [
            "C1=CC=CC=C1",  # benzene ring
            "C1CCC2CCCCC2C1",  # fused bicyclic
            "C1CC2CCC1CC2",   # bridged bicyclic
            "C1CCC2=CC=CC=C2C1"  # tetralin-type
        ]
        
        for product in products:
            # Check for 6-membered rings (common in Diels-Alder products)
            ring_info = product.GetRingInfo()
            six_membered_rings = [ring for ring in ring_info.AtomRings() if len(ring) == 6]
            
            if six_membered_rings:
                # Check if product has bicyclic character
                if len(Chem.GetSymmSSSR(product)) >= 2:
                    return True
                
                # Alternative: check for specific bicyclic substructures
                for pattern in bicyclic_patterns:
                    try:
                        pattern_mol = Chem.MolFromSmarts(pattern)
                        if pattern_mol and product.HasSubstructMatch(pattern_mol):
                            return True
                    except:
                        continue
        
        return False
