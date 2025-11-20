"""Generated evaluation code for: Late stage pyridine ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageRingFormation(BaseScoring):
    """
    Evaluates whether a specific ring is formed late in the synthesis route.
    Uses atom mapping to detect when the target ring structure appears in products
    but the ring connectivity is broken across multiple reactants.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]  # "late" or "early"
        self.direction = config["parameters"]["direction"]  # "formation" or "break"
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        
        if self.timing == "late":
            return 1 - x  # Later formation gets higher score
        else:  # early
            return x  # Earlier formation gets higher score
            
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction step involves formation of the target ring.
        Returns True if the ring is present in product but ring connectivity
        is broken across reactants.
        """
        if self.direction != "formation":
            return False
            
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            product_smiles, reactant_smiles = mapped_rxn.split(">>")
            product_mol = Chem.MolFromSmiles(product_smiles)
            reactant_mols = [Chem.MolFromSmiles(r) for r in reactant_smiles.split(".")]
            
            if not product_mol or not all(reactant_mols):
                return False
                
            # Check if target ring is present in product
            if not product_mol.HasSubstructMatch(self.ring_pattern):
                return False
                
            # Get the ring atoms in the product using atom mapping
            ring_matches = product_mol.GetSubstructMatches(self.ring_pattern)
            
            for ring_match in ring_matches:
                ring_atom_maps = []
                for atom_idx in ring_match:
                    atom = product_mol.GetAtomWithIdx(atom_idx)
                    if atom.GetAtomMapNum() > 0:
                        ring_atom_maps.append(atom.GetAtomMapNum())
                
                if len(ring_atom_maps) < 3:  # Need sufficient mapped atoms to verify
                    continue
                    
                # Check if ring connectivity is broken in reactants
                if self._is_ring_connectivity_broken(ring_atom_maps, reactant_mols):
                    return True
                    
            return False
            
        except Exception:
            return False
            
    def _is_ring_connectivity_broken(self, ring_atom_maps, reactant_mols):
        """
        Check if the ring atoms are distributed across reactants or
        present in one reactant but not forming the complete ring structure.
        """
        # Map atom map numbers to reactant molecules
        atom_to_reactant = {}
        for reactant_idx, reactant in enumerate(reactant_mols):
            for atom in reactant.GetAtoms():
                if atom.GetAtomMapNum() in ring_atom_maps:
                    atom_to_reactant[atom.GetAtomMapNum()] = reactant_idx
                    
        if len(atom_to_reactant) < 3:
            return False
            
        # Check if ring atoms are split across multiple reactants
        reactant_indices = set(atom_to_reactant.values())
        if len(reactant_indices) > 1:
            return True
            
        # If all ring atoms are in one reactant, check if they form the ring there
        reactant_with_atoms = reactant_indices.pop()
        reactant_mol = reactant_mols[reactant_with_atoms]
        
        # If the reactant doesn't contain the complete ring pattern, 
        # then the ring is being formed in this step
        return not reactant_mol.HasSubstructMatch(self.ring_pattern)
