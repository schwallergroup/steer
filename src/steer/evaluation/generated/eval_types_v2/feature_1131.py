"""Generated evaluation code for: Late stage cyclopropane spiro ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageCyclopropaneSpiroFormation(BaseScoring):
    """
    Evaluates late-stage cyclopropane spiro ring formation in synthesis routes.
    Checks for the formation of spiro-cyclopropane rings and rewards when this
    occurs later in the synthesis sequence.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "continuous")
        self.target_depth = config.get("target_depth", {}).get("value", 0.8)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Spiro-cyclopropane formation doesn't happen
        else:
            # Late-stage formation is rewarded (higher depth fraction is better)
            if self.condition_type == "bool":
                return 1 if x >= self.target_depth else 0
            else:
                # Reward formations that occur late in the sequence
                return max(0, min(10, x * 10))
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction step forms a spiro-cyclopropane ring.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles")
        
        if not mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            product = Chem.MolFromSmiles(rxn_parts[0])
            reactants = [Chem.MolFromSmiles(r) for r in rxn_parts[1].split(".") if r]
            
            if not product or not all(reactants):
                return False
            
            # Check if product has spiro-cyclopropane that reactants don't have
            product_spiro_cp = self._count_spiro_cyclopropanes(product)
            reactant_spiro_cp = sum(self._count_spiro_cyclopropanes(r) for r in reactants)
            
            return product_spiro_cp > reactant_spiro_cp
            
        except:
            return False
    
    def _count_spiro_cyclopropanes(self, mol) -> int:
        """
        Count spiro-cyclopropane rings in a molecule.
        A spiro-cyclopropane has one carbon shared between a 3-membered ring and another ring.
        """
        if not mol:
            return 0
            
        # Find all 3-membered rings (cyclopropanes)
        ring_info = mol.GetRingInfo()
        three_rings = [ring for ring in ring_info.AtomRings() if len(ring) == 3]
        
        if not three_rings:
            return 0
        
        spiro_count = 0
        
        for three_ring in three_rings:
            # Check each carbon in the cyclopropane ring
            for atom_idx in three_ring:
                atom = mol.GetAtomWithIdx(atom_idx)
                if atom.GetSymbol() == 'C':
                    # Check if this carbon is part of another ring (spiro center)
                    atom_rings = [ring for ring in ring_info.AtomRings() if atom_idx in ring]
                    
                    # Spiro center should be in at least 2 rings (the cyclopropane and another)
                    if len(atom_rings) >= 2:
                        # Verify one ring is the cyclopropane and others are different
                        other_rings = [ring for ring in atom_rings if ring != tuple(three_ring)]
                        if other_rings:
                            spiro_count += 1
                            break  # Only count once per cyclopropane ring
        
        return spiro_count
