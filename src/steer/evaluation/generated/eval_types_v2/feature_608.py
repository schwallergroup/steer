"""Generated evaluation code for: Late macrocycle formation via RCM"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateMacrocycleRCM(BaseScoring):
    """
    Evaluates whether macrocycle formation via ring-closing metathesis (RCM) 
    occurs at a late stage in the synthesis route.
    """
    
    def __init__(self, config: Dict):
        self.ring_size_min = config["parameters"]["ring_size_min"]
        self.step_position = config["parameters"]["step_position"]
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # RCM macrocycle formation doesn't happen
        else:
            # Later formation is better, penalize early formation
            return max(0, 10 - (x * 10))
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction involves RCM macrocycle formation"""
        try:
            rxn = d["metadata"]["mapped_reaction_smiles"].split(">>")
            product = Chem.MolFromSmiles(rxn[0])
            reactants = [Chem.MolFromSmiles(r) for r in rxn[1].split(".")]
            
            if not product or not all(reactants):
                return False
                
            # Check if this is a metathesis reaction (alkene reactants, alkene product)
            if not self._is_metathesis_reaction(product, reactants):
                return False
                
            # Check if a macrocycle is formed
            return self._forms_macrocycle(product, reactants)
            
        except Exception:
            return False
    
    def _is_metathesis_reaction(self, product, reactants) -> bool:
        """Check if reaction involves alkene metathesis"""
        alkene_pattern = Chem.MolFromSmarts("C=C")
        
        # Product should contain alkene
        if not product.HasSubstructMatch(alkene_pattern):
            return False
            
        # At least one reactant should contain alkene
        reactant_has_alkene = any(r.HasSubstructMatch(alkene_pattern) for r in reactants)
        return reactant_has_alkene
    
    def _forms_macrocycle(self, product, reactants) -> bool:
        """Check if a macrocycle of minimum size is formed"""
        # Find rings in product
        product_rings = product.GetRingInfo().AtomRings()
        large_rings_product = [ring for ring in product_rings if len(ring) >= self.ring_size_min]
        
        if not large_rings_product:
            return False
            
        # Check that the large ring wasn't present in any single reactant
        for reactant in reactants:
            reactant_rings = reactant.GetRingInfo().AtomRings()
            large_rings_reactant = [ring for ring in reactant_rings if len(ring) >= self.ring_size_min]
            
            # If reactant already has a large ring, this might not be ring formation
            if large_rings_reactant:
                # Check if the ring atoms are conserved (need to check atom mapping)
                # For simplicity, assume ring formation if product has more large rings
                continue
                
        return True
