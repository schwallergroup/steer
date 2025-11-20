"""Generated evaluation code for: Late stage complex ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageComplexRingFormation(BaseScoring):
    """
    Evaluates whether complex fused ring formation occurs at late stages of synthesis.
    Looks for formation of fused ring systems (like purine-like structures) within
    a specified depth threshold from the target molecule.
    """
    
    def __init__(self, config: Dict):
        self.depth_threshold = config["parameters"]["depth_threshold"]
        self.ring_type = config["parameters"]["ring_type"]
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # No complex ring formation found
        
        # Convert depth to fraction (x is already normalized depth)
        if x <= self.depth_threshold / 10.0:  # Within late stage threshold
            return 10 * (1 - x)  # Earlier formation gets higher score
        else:
            return 2 * (1 - x)  # Formation too early, lower score
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction involves complex fused ring formation"""
        try:
            rxn = d["metadata"]["mapped_reaction_smiles"].split(">>")
            product = Chem.MolFromSmiles(rxn[0])
            reactants = [Chem.MolFromSmiles(r) for r in rxn[1].split(".")]
            
            if not product or not all(reactants):
                return False
                
            # Get ring systems in product and reactants
            product_fused_rings = self._count_fused_ring_systems(product)
            reactant_fused_rings = sum(self._count_fused_ring_systems(r) for r in reactants)
            
            # Check if new fused ring system is formed
            if product_fused_rings > reactant_fused_rings:
                # Verify it's a complex ring formation (purine-like or similar)
                return self._has_complex_ring_formation(product, reactants)
                
        except Exception:
            pass
            
        return False
    
    def _count_fused_ring_systems(self, mol) -> int:
        """Count number of fused ring systems in molecule"""
        if not mol:
            return 0
            
        ring_info = mol.GetRingInfo()
        rings = ring_info.AtomRings()
        
        if len(rings) < 2:
            return len(rings)
        
        # Build adjacency of rings (rings that share atoms)
        ring_adjacency = []
        for i, ring1 in enumerate(rings):
            adjacent = []
            for j, ring2 in enumerate(rings):
                if i != j and len(set(ring1) & set(ring2)) > 0:
                    adjacent.append(j)
            ring_adjacency.append(adjacent)
        
        # Count connected components (fused systems)
        visited = [False] * len(rings)
        fused_systems = 0
        
        for i in range(len(rings)):
            if not visited[i]:
                self._dfs_rings(i, ring_adjacency, visited)
                fused_systems += 1
                
        return fused_systems
    
    def _dfs_rings(self, ring_idx, adjacency, visited):
        """DFS helper for finding connected ring systems"""
        visited[ring_idx] = True
        for neighbor in adjacency[ring_idx]:
            if not visited[neighbor]:
                self._dfs_rings(neighbor, adjacency, visited)
    
    def _has_complex_ring_formation(self, product, reactants) -> bool:
        """Check if the ring formation creates complex heterocycles like purines"""
        # Purine-like pattern (fused 6-5 ring with nitrogens)
        purine_pattern = Chem.MolFromSmarts("c1ncnc2[nH]cnc12")
        pyrimidine_purine_pattern = Chem.MolFromSmarts("c1ncnc2ncnc12")
        
        # General fused heterocycle patterns
        fused_hetero_patterns = [
            Chem.MolFromSmarts("c1ccc2ccccc12"),  # Fused aromatic
            Chem.MolFromSmarts("c1ccc2ncncc12"),  # Quinoxaline-like
            Chem.MolFromSmarts("c1cnc2ncncc12"),  # Pteridine-like
            Chem.MolFromSmarts("n1cnc2ncnc12"),   # Purine-like
        ]
        
        # Check if product has complex fused ring
        has_complex_product = (purine_pattern and product.HasSubstructMatch(purine_pattern)) or \
                             (pyrimidine_purine_pattern and product.HasSubstructMatch(pyrimidine_purine_pattern)) or \
                             any(pattern and product.HasSubstructMatch(pattern) for pattern in fused_hetero_patterns)
        
        if not has_complex_product:
            return False
            
        # Check that reactants don't already have this complex system
        for reactant in reactants:
            if (purine_pattern and reactant.HasSubstructMatch(purine_pattern)) or \
               (pyrimidine_purine_pattern and reactant.HasSubstructMatch(pyrimidine_purine_pattern)):
                return False
                
        return True
