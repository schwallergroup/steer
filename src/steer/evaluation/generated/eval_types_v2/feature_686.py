"""Generated evaluation code for: Late stage tricyclic core formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageTricyclicFormation(BaseScoring):
    """
    Evaluates whether a tricyclic core formation occurs at a late stage in the synthesis.
    Checks for ring formation reactions that create tricyclic systems after the specified
    stage threshold (default 0.8, meaning in the last 20% of the synthesis).
    """
    
    def __init__(self, config: Dict):
        self.ring_count = config.get("ring_count", 1)
        self.stage_threshold = config.get("stage_threshold", 0.8)
        self.timing = config.get("timing", "late")
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Tricyclic formation doesn't happen
        
        if self.timing == "late":
            if x >= self.stage_threshold:
                return 10  # Perfect score for very late stage formation
            else:
                # Penalize earlier formation, scale from 0-10 based on how close to threshold
                return max(0, (x - 0.5) / (self.stage_threshold - 0.5) * 10)
        else:
            # For other timing preferences, invert the scoring
            return max(0, 10 * (1 - x))
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction forms a tricyclic system"""
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            
            if not all(reactants) or not all(products):
                return False
            
            # Count rings in reactants and products
            reactant_ring_counts = [self._count_rings_in_fused_systems(mol) for mol in reactants]
            product_ring_counts = [self._count_rings_in_fused_systems(mol) for mol in products]
            
            max_reactant_rings = max(reactant_ring_counts) if reactant_ring_counts else 0
            max_product_rings = max(product_ring_counts) if product_ring_counts else 0
            
            # Check if we formed a tricyclic system (3+ rings) and increased ring count
            rings_formed = max_product_rings - max_reactant_rings
            
            return (max_product_rings >= 3 and 
                   rings_formed >= self.ring_count and
                   self._is_tricyclic_formation(reactants, products))
            
        except Exception:
            return False
    
    def _count_rings_in_fused_systems(self, mol) -> int:
        """Count the maximum number of rings in any fused ring system"""
        if mol is None:
            return 0
        
        ring_info = mol.GetRingInfo()
        rings = ring_info.AtomRings()
        
        if not rings:
            return 0
        
        # Find connected ring systems
        ring_systems = []
        for ring in rings:
            ring_set = set(ring)
            merged = False
            
            for i, system in enumerate(ring_systems):
                if ring_set & system:  # If rings share atoms (fused)
                    ring_systems[i] = system | ring_set
                    merged = True
                    break
            
            if not merged:
                ring_systems.append(ring_set)
        
        # Count rings in each system
        system_ring_counts = []
        for system in ring_systems:
            count = 0
            for ring in rings:
                if set(ring) & system:
                    count += 1
            system_ring_counts.append(count)
        
        return max(system_ring_counts) if system_ring_counts else 0
    
    def _is_tricyclic_formation(self, reactants, products) -> bool:
        """Additional check to ensure we're forming a true tricyclic core"""
        # Look for common tricyclic patterns in products
        tricyclic_patterns = [
            # Purine-like tricyclic cores
            "c1nc2[nH]cnc2[nH]1",  # Purine core
            "c1nc2ncnc2[nH]1",     # Alternative purine
            # Other common tricyclic systems
            "C1CC2CCC3CCCC(C1)C23", # Tricyclic saturated
            "c1cc2cc3ccccc3cc2cc1",  # Anthracene-like
        ]
        
        for product in products:
            for pattern in tricyclic_patterns:
                try:
                    pattern_mol = Chem.MolFromSmarts(pattern)
                    if pattern_mol and product.HasSubstructMatch(pattern_mol):
                        # Verify this pattern wasn't already present in reactants
                        for reactant in reactants:
                            if reactant.HasSubstructMatch(pattern_mol):
                                return False  # Pattern already existed
                        return True
                except:
                    continue
        
        return False
