"""Generated evaluation code for: Early Friedel-Crafts acylation for ring closure"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlyFriedelCraftsRingClosure(BaseScoring):
    """
    Evaluates routes for early intramolecular Friedel-Crafts acylation leading to 6-membered ring closure.
    Detects ketone formation adjacent to aromatic rings via intramolecular acylation.
    """
    
    def __init__(self, config: Dict):
        self.timing = config["parameters"]["timing"]  # "early"
        self.ring_size = config["parameters"]["ring_size"]  # 6
        self.target_early_depth = 0.8  # Earlier than 80% of route depth is considered "early"
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Reaction doesn't occur in route
        
        if self.timing == "early":
            # Reward early occurrence (lower depth fraction)
            if x <= self.target_early_depth:
                return 10 - (x * 5)  # Score 10 at depth 0, 6 at depth 0.8
            else:
                return max(0, 5 - (x - self.target_early_depth) * 10)  # Penalty for late occurrence
        else:
            return 5  # Neutral score if timing not specified
    
    def hit_condition(self, d):
        """
        Detects intramolecular Friedel-Crafts acylation forming 6-membered rings.
        Looks for aromatic ketone formation where the carbonyl and aromatic ring
        were previously connected by a chain that gets cyclized.
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse molecules
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            
            if not all(products) or not all(reactants):
                return False
            
            # Look for aromatic ketone formation (Friedel-Crafts acylation signature)
            aromatic_ketone_pattern = Chem.MolFromSmarts("[cR:1]-[CR:2](=[OR:3])")
            
            # Check if products contain new aromatic ketones that weren't in reactants
            for prod in products:
                matches = prod.GetSubstructMatches(aromatic_ketone_pattern)
                for match in matches:
                    aromatic_atom, carbonyl_atom, oxygen_atom = match
                    
                    # Verify this is part of a 6-membered ring involving the ketone
                    if self._is_six_membered_ring_ketone(prod, aromatic_atom, carbonyl_atom):
                        # Check if this ketone was formed in this reaction (not pre-existing)
                        if not self._ketone_exists_in_reactants(reactants, aromatic_atom, carbonyl_atom, rxn_smiles):
                            # Verify intramolecular nature by checking atom mapping
                            if self._is_intramolecular_cyclization(rxn_smiles, aromatic_atom, carbonyl_atom):
                                return True
            
            return False
            
        except Exception:
            return False
    
    def _is_six_membered_ring_ketone(self, mol, aromatic_atom_idx, carbonyl_atom_idx):
        """Check if the aromatic ketone is part of a 6-membered ring system."""
        try:
            # Get ring info
            ring_info = mol.GetRingInfo()
            atom_rings = ring_info.AtomRings()
            
            # Find rings containing the carbonyl carbon
            for ring in atom_rings:
                if carbonyl_atom_idx in ring and len(ring) == self.ring_size:
                    # Check if aromatic atom is also in the ring or adjacent
                    if aromatic_atom_idx in ring:
                        return True
                    # Check if aromatic atom is adjacent to any ring atom
                    aromatic_atom = mol.GetAtomWithIdx(aromatic_atom_idx)
                    for neighbor in aromatic_atom.GetNeighbors():
                        if neighbor.GetIdx() in ring:
                            return True
            return False
        except Exception:
            return False
    
    def _ketone_exists_in_reactants(self, reactants, aromatic_map, carbonyl_map, rxn_smiles):
        """Check if the aromatic ketone already exists in reactants using atom mapping."""
        try:
            aromatic_ketone_pattern = Chem.MolFromSmarts("[cR:1]-[CR:2](=[OR:3])")
            
            for reactant in reactants:
                matches = reactant.GetSubstructMatches(aromatic_ketone_pattern)
                for match in matches:
                    # Get atom map numbers
                    r_aromatic = reactant.GetAtomWithIdx(match[0]).GetAtomMapNum()
                    r_carbonyl = reactant.GetAtomWithIdx(match[1]).GetAtomMapNum()
                    
                    if r_aromatic == aromatic_map and r_carbonyl == carbonyl_map:
                        return True
            return False
        except Exception:
            return False
    
    def _is_intramolecular_cyclization(self, rxn_smiles, aromatic_map, carbonyl_map):
        """Verify this is intramolecular by checking if both atoms come from same reactant."""
        try:
            reactants_smiles = rxn_smiles.split(">>")[0]
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            
            for reactant in reactants:
                aromatic_found = False
                carbonyl_precursor_found = False
                
                for atom in reactant.GetAtoms():
                    map_num = atom.GetAtomMapNum()
                    if map_num == aromatic_map:
                        aromatic_found = True
                    elif map_num == carbonyl_map:
                        carbonyl_precursor_found = True
                
                # If both atoms are in the same reactant, it's intramolecular
                if aromatic_found and carbonyl_precursor_found:
                    return True
            
            return False
        except Exception:
            return False
