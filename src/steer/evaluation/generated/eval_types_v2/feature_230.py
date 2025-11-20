"""Generated evaluation code for: Late stage intramolecular cyclization for ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageIntramolecularCyclization(BaseScoring):
    """
    Evaluates synthesis routes for late-stage intramolecular cyclization reactions
    that form rings of a specified size. Rewards cyclizations that occur at the
    target depth in the synthesis tree.
    """
    
    def __init__(self, config: Dict):
        self.target_step = config["parameters"]["ring_formation_step"]
        self.total_steps = config["parameters"]["total_steps"]
        self.formation_type = config["parameters"]["formation_type"]
        self.ring_size = config["parameters"]["ring_size"]
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # No intramolecular cyclization found
        
        # Convert depth fraction to actual step number
        actual_step = int(x * self.total_steps) + 1
        
        # Score based on how close to target step
        step_difference = abs(actual_step - self.target_step)
        
        # Best score (10) for exact match, decreasing with distance
        if step_difference == 0:
            return 10
        elif step_difference <= 1:
            return 8
        elif step_difference <= 2:
            return 6
        elif step_difference <= 3:
            return 4
        else:
            return 2
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction node represents an intramolecular cyclization
        that forms a ring of the specified size.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            product_smiles = rxn_parts[0]
            reactant_smiles = rxn_parts[1]
            
            product_mol = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r) for r in reactant_smiles.split(".")]
            
            if not product_mol or not all(reactants):
                return False
                
            # Check if it's intramolecular (single reactant forms ring)
            if self.formation_type == "intramolecular" and len(reactants) != 1:
                return False
                
            # Check if a new ring of specified size was formed
            return self._detect_ring_formation(product_mol, reactants[0], self.ring_size)
            
        except Exception:
            return False
    
    def _detect_ring_formation(self, product, reactant, target_ring_size):
        """
        Detect if a new ring of the specified size was formed by comparing
        ring systems in product vs reactant.
        """
        if not product or not reactant:
            return False
            
        # Get ring information for both molecules
        product_rings = product.GetRingInfo().AtomRings()
        reactant_rings = reactant.GetRingInfo().AtomRings()
        
        # Count rings of target size
        product_target_rings = sum(1 for ring in product_rings if len(ring) == target_ring_size)
        reactant_target_rings = sum(1 for ring in reactant_rings if len(ring) == target_ring_size)
        
        # Check if we gained at least one ring of the target size
        if product_target_rings > reactant_target_rings:
            # Additional check: verify the new ring uses existing atoms (intramolecular)
            return self._is_intramolecular_cyclization(product, reactant)
            
        return False
    
    def _is_intramolecular_cyclization(self, product, reactant):
        """
        Verify that the cyclization is intramolecular by checking that
        no new heavy atoms were added during ring formation.
        """
        product_heavy_count = product.GetNumHeavyAtoms()
        reactant_heavy_count = reactant.GetNumHeavyAtoms()
        
        # For intramolecular cyclization, heavy atom count should be same or decrease
        # (may decrease if leaving groups are eliminated)
        return product_heavy_count <= reactant_heavy_count
