"""Generated evaluation code for: Early Diels-Alder bicyclic core formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlyDielsAlderBicyclicCore(BaseScoring):
    """
    Evaluates whether a Diels-Alder reaction that creates multiple rings occurs early in the synthesis route.
    
    This class detects [4+2] cycloaddition reactions that form bicyclic structures and scores
    routes based on how early this key ring-forming reaction occurs.
    """
    
    def __init__(self, config: Dict):
        self.early_threshold = config.get("early_threshold", 0.3)  # Early = first 30% of route
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Diels-Alder bicyclic formation doesn't happen
        
        # Score based on how early it occurs (lower depth fraction = higher score)
        if x <= self.early_threshold:
            return 10  # Perfect score for very early occurrence
        else:
            # Linearly decrease score as depth increases
            return max(0, 10 * (1 - x))
    
    def hit_condition(self, d) -> bool:
        """
        Detects if a reaction is a Diels-Alder that creates multiple rings.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            reactants_smiles, product_smiles = mapped_rxn.split(">>")
            
            # Parse molecules
            product = Chem.MolFromSmiles(product_smiles.strip())
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            
            if not product or not all(reactant_mols):
                return False
            
            # Count rings in reactants vs product
            reactant_ring_count = sum(mol.GetRingInfo().NumRings() for mol in reactant_mols)
            product_ring_count = product.GetRingInfo().NumRings()
            
            # Must create multiple rings (at least 2 new rings)
            rings_formed = product_ring_count - reactant_ring_count
            if rings_formed < 2:
                return False
            
            # Check for Diels-Alder pattern: should have exactly 2 reactants
            if len(reactant_mols) != 2:
                return False
                
            # Look for characteristic Diels-Alder patterns
            return self._is_diels_alder_reaction(reactant_mols, product)
            
        except Exception:
            return False
    
    def _is_diels_alder_reaction(self, reactants, product):
        """
        Checks if the reaction matches Diels-Alder characteristics.
        """
        # Common Diels-Alder diene patterns (conjugated systems)
        diene_patterns = [
            "C=CC=C",  # Simple butadiene
            "c1ccccc1C=CC=C",  # Aromatic diene
            "C=CC=CC=C",  # Extended diene
        ]
        
        # Common dienophile patterns (electron-deficient alkenes)
        dienophile_patterns = [
            "C=C",  # Simple alkene
            "C=CC=O",  # α,β-unsaturated carbonyl
            "C=CC(=O)",  # α,β-unsaturated carbonyl variant
            "C=CC#N",  # α,β-unsaturated nitrile
        ]
        
        # Check if we have one diene and one dienophile
        has_diene = False
        has_dienophile = False
        
        for reactant in reactants:
            # Check for diene patterns
            for pattern in diene_patterns:
                if reactant.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                    has_diene = True
                    break
            
            # Check for dienophile patterns
            for pattern in dienophile_patterns:
                if reactant.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                    has_dienophile = True
                    break
        
        # Must have both diene and dienophile, and product should have 6-membered ring
        six_membered_rings = len([ring for ring in product.GetRingInfo().AtomRings() if len(ring) == 6])
        
        return has_diene and has_dienophile and six_membered_rings >= 1
