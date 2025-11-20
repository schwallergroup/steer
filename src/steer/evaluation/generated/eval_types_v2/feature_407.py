"""Generated evaluation code for: Late stage intramolecular C-N cyclization"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageIntramolecularCNCyclization(BaseScoring):
    """
    Evaluates routes for late-stage intramolecular C-N cyclization forming 6-membered rings.
    Specifically targets pyrido[2,3-d]pyrimidine-like ring formations via C-N bond formation.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "linear")
        self.target_depth = config.get("target_depth", {}).get("value", 0.9)  # Late stage preferred
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # No cyclization found
        else:
            # Reward later cyclization (closer to 1.0 depth fraction)
            if self.condition_type == "bool":
                return 1 if x > 0.7 else 0  # Must be in final 30% of route
            else:
                # Linear scoring - later is better
                return max(0, min(10, 10 * x))
    
    def hit_condition(self, d):
        """Check if this reaction performs intramolecular C-N cyclization forming a 6-membered ring"""
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
                
            reactant_smiles, product_smiles = rxn_smiles.split(">>")
            
            # Parse molecules
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactant_smiles.split(".")]
            product = Chem.MolFromSmiles(product_smiles.strip())
            
            if not all(reactants) or not product:
                return False
            
            # Check if this is an intramolecular reaction (single reactant to product)
            if len(reactants) != 1:
                return False
                
            reactant = reactants[0]
            
            # Count 6-membered rings in reactant vs product
            reactant_6rings = self._count_6membered_rings(reactant)
            product_6rings = self._count_6membered_rings(product)
            
            # Must form exactly one new 6-membered ring
            if product_6rings != reactant_6rings + 1:
                return False
            
            # Check for C-N bond formation in the new ring
            return self._has_cn_ring_formation(reactant, product)
            
        except Exception:
            return False
    
    def _count_6membered_rings(self, mol):
        """Count 6-membered rings in molecule"""
        if not mol:
            return 0
        ring_info = mol.GetRingInfo()
        return len([ring for ring in ring_info.AtomRings() if len(ring) == 6])
    
    def _has_cn_ring_formation(self, reactant, product):
        """Check if a C-N bond is formed in a 6-membered ring"""
        try:
            # Get atom mapping for tracking atoms
            reactant_map = {}
            product_map = {}
            
            for atom in reactant.GetAtoms():
                map_num = atom.GetAtomMapNum()
                if map_num > 0:
                    reactant_map[map_num] = atom.GetIdx()
                    
            for atom in product.GetAtoms():
                map_num = atom.GetAtomMapNum()
                if map_num > 0:
                    product_map[map_num] = atom.GetIdx()
            
            # Find new bonds in product
            reactant_bonds = set()
            for bond in reactant.GetBonds():
                begin_map = bond.GetBeginAtom().GetAtomMapNum()
                end_map = bond.GetEndAtom().GetAtomMapNum()
                if begin_map > 0 and end_map > 0:
                    reactant_bonds.add(tuple(sorted([begin_map, end_map])))
            
            # Check product bonds for new C-N bonds in 6-membered rings
            ring_info = product.GetRingInfo()
            six_rings = [ring for ring in ring_info.AtomRings() if len(ring) == 6]
            
            for bond in product.GetBonds():
                begin_atom = bond.GetBeginAtom()
                end_atom = bond.GetEndAtom()
                begin_map = begin_atom.GetAtomMapNum()
                end_map = end_atom.GetAtomMapNum()
                
                # Skip if atoms don't have mapping
                if begin_map == 0 or end_map == 0:
                    continue
                    
                bond_key = tuple(sorted([begin_map, end_map]))
                
                # Check if this is a new bond
                if bond_key not in reactant_bonds:
                    # Check if it's a C-N bond
                    atoms = [begin_atom, end_atom]
                    symbols = [atom.GetSymbol() for atom in atoms]
                    
                    if set(symbols) == {'C', 'N'}:
                        # Check if both atoms are in a 6-membered ring
                        begin_idx = product_map.get(begin_map)
                        end_idx = product_map.get(end_map)
                        
                        if begin_idx is not None and end_idx is not None:
                            for ring in six_rings:
                                if begin_idx in ring and end_idx in ring:
                                    return True
            
            return False
            
        except Exception:
            return False
