"""Generated evaluation code for: Early spiro ring system construction"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlySpiroRingConstruction(BaseScoring):
    """
    Checks if spiro ring system construction occurs early in the synthesis route.
    A spiro center is defined as a quaternary carbon atom shared by exactly two rings.
    """
    
    def __init__(self, config: Dict):
        self.stage_threshold = config["parameters"]["stage_threshold"]
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Spiro construction doesn't happen
        else:
            # Early construction (lower depth fraction) gets higher score
            return 1 - x
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction forms a spiro ring system."""
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
                
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse molecules
            products = [Chem.MolFromSmiles(mol) for mol in products_smiles.split(".")]
            reactants = [Chem.MolFromSmiles(mol) for mol in reactants_smiles.split(".")]
            
            if not all(products) or not all(reactants):
                return False
            
            # Count spiro centers in reactants and products
            reactant_spiro_count = sum(self._count_spiro_centers(mol) for mol in reactants)
            product_spiro_count = sum(self._count_spiro_centers(mol) for mol in products)
            
            # Check if spiro centers increased (spiro formation)
            return product_spiro_count > reactant_spiro_count
            
        except Exception:
            return False
    
    def _count_spiro_centers(self, mol) -> int:
        """Count the number of spiro centers in a molecule."""
        if mol is None:
            return 0
            
        try:
            # Get ring information
            ring_info = mol.GetRingInfo()
            atom_rings = ring_info.AtomRings()
            
            spiro_count = 0
            
            # Check each atom
            for atom in mol.GetAtoms():
                # Only consider carbon atoms
                if atom.GetAtomicNum() != 6:
                    continue
                    
                # Check if atom is in exactly 2 rings
                atom_idx = atom.GetIdx()
                rings_containing_atom = [ring for ring in atom_rings if atom_idx in ring]
                
                if len(rings_containing_atom) == 2:
                    # Check if it's a true spiro center (quaternary carbon)
                    if atom.GetDegree() == 4:
                        # Verify the two rings don't share any other atoms
                        ring1, ring2 = rings_containing_atom
                        shared_atoms = set(ring1) & set(ring2)
                        
                        # Only the spiro center should be shared
                        if len(shared_atoms) == 1 and atom_idx in shared_atoms:
                            spiro_count += 1
                            
            return spiro_count
            
        except Exception:
            return 0
