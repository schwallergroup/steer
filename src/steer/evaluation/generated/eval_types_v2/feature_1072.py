"""Generated evaluation code for: Late stage ring closing metathesis macrocyclization"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageRCMMacrocyclization(BaseScoring):
    """
    Evaluates routes for late-stage ring-closing metathesis (RCM) macrocyclization.
    Checks if a macrocyclic ring (8+ membered) is formed via RCM at a specified depth.
    """
    
    def __init__(self, config: Dict):
        self.target_depth = config["parameters"]["ring_formation_depth"]
        self.min_ring_size = 8  # Minimum size for macrocycle
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # RCM macrocyclization doesn't happen
        else:
            # Late-stage formation is better, penalize early formation
            depth_penalty = abs(x - (self.target_depth / 10.0))
            return max(0, 1 - depth_penalty)
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction is an RCM macrocyclization"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        rxn_parts = mapped_rxn.split(">>")
        product_smiles = rxn_parts[0]
        reactant_smiles = rxn_parts[1]
        
        try:
            product_mol = Chem.MolFromSmiles(product_smiles)
            reactant_mols = [Chem.MolFromSmiles(r) for r in reactant_smiles.split(".")]
            
            if not product_mol or not all(reactant_mols):
                return False
                
            # Check if this is RCM: reactant has alkene groups that form a ring in product
            if not self._has_rcm_pattern(reactant_mols, product_mol):
                return False
                
            # Check if a new macrocycle was formed
            return self._forms_new_macrocycle(reactant_mols, product_mol)
            
        except:
            return False
    
    def _has_rcm_pattern(self, reactants, product):
        """Check for RCM pattern: alkene metathesis reaction"""
        # Look for alkene pattern in reactants and product
        alkene_pattern = Chem.MolFromSmarts("[C]=[C]")
        
        # Product should have alkenes
        if not product.HasSubstructMatch(alkene_pattern):
            return False
            
        # At least one reactant should have multiple alkenes
        for reactant in reactants:
            alkene_matches = reactant.GetSubstructMatches(alkene_pattern)
            if len(alkene_matches) >= 2:
                return True
                
        return False
    
    def _forms_new_macrocycle(self, reactants, product):
        """Check if a new macrocycle (8+ membered ring) is formed"""
        # Get ring info for product and reactants
        product_rings = self._get_large_rings(product)
        
        # Get all rings from reactants
        reactant_rings = set()
        for reactant in reactants:
            reactant_rings.update(self._get_large_rings(reactant))
        
        # Check if product has a macrocycle not present in reactants
        new_macrocycles = product_rings - reactant_rings
        
        return len(new_macrocycles) > 0
    
    def _get_large_rings(self, mol):
        """Get set of large rings (8+ members) represented by their atom sets"""
        large_rings = set()
        ring_info = mol.GetRingInfo()
        
        for ring in ring_info.AtomRings():
            if len(ring) >= self.min_ring_size:
                # Use atom map numbers if available, otherwise atom indices
                atom_ids = []
                for atom_idx in ring:
                    atom = mol.GetAtomWithIdx(atom_idx)
                    map_num = atom.GetAtomMapNum()
                    atom_ids.append(map_num if map_num > 0 else atom_idx)
                large_rings.add(tuple(sorted(atom_ids)))
                
        return large_rings
