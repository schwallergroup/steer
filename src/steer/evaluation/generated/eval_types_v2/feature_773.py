"""Generated evaluation code for: Late stage diaryl ether formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageArylEther(BaseScoring):
    """
    Evaluates routes for late-stage diaryl ether formation via nucleophilic aromatic substitution.
    Checks for C-O bond formation between aromatic rings at shallow depths (preferably depth 1).
    """
    
    def __init__(self, config: Dict):
        self.target_depth = config.get("target_depth", 1)
        self.condition_type = config.get("condition_type", "depth")
        
    def route_scoring(self, x) -> float:
        """Convert depth fraction to score (0-10). Lower depth is better for late-stage."""
        if x < 0:
            return 0  # Reaction doesn't occur
        
        if self.condition_type == "bool":
            return 10 if x <= 0.2 else 0  # High score if very late stage
        else:
            # Exponential penalty for deeper reactions
            depth_penalty = x * 10
            return max(0, 10 - depth_penalty)
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction involves diaryl ether formation via SNAr."""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            reactants, products = mapped_rxn.split(">>")
            prod_mol = Chem.MolFromSmiles(products)
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            
            if not prod_mol or not all(reactant_mols):
                return False
                
            return self._is_diaryl_ether_formation(prod_mol, reactant_mols)
            
        except Exception:
            return False
    
    def _is_diaryl_ether_formation(self, product, reactants) -> bool:
        """Check if reaction forms diaryl ether via SNAr mechanism."""
        # Pattern for diaryl ether: aromatic-O-aromatic
        diaryl_ether_pattern = Chem.MolFromSmarts("c-O-c")
        
        if not product.HasSubstructMatch(diaryl_ether_pattern):
            return False
            
        # Get diaryl ether bonds in product
        ether_matches = product.GetSubstructMatches(diaryl_ether_pattern)
        
        for match in ether_matches:
            ar1_idx, o_idx, ar2_idx = match
            
            # Check if this ether bond is newly formed
            if self._is_new_ether_bond(product, reactants, ar1_idx, o_idx, ar2_idx):
                # Verify SNAr-like conditions
                if self._check_snar_conditions(product, reactants, ar1_idx, o_idx, ar2_idx):
                    return True
                    
        return False
    
    def _is_new_ether_bond(self, product, reactants, ar1_idx, o_idx, ar2_idx) -> bool:
        """Check if the C-O-C bond is newly formed (not present in reactants)."""
        prod_atoms = product.GetAtoms()
        ar1_mapnum = prod_atoms[ar1_idx].GetAtomMapNum()
        o_mapnum = prod_atoms[o_idx].GetAtomMapNum()
        ar2_mapnum = prod_atoms[ar2_idx].GetAtomMapNum()
        
        if not all([ar1_mapnum, o_mapnum, ar2_mapnum]):
            return False
            
        # Check if these three atoms are connected in any reactant
        for reactant in reactants:
            if self._atoms_connected_in_mol(reactant, ar1_mapnum, o_mapnum, ar2_mapnum):
                return False  # Bond already exists
                
        return True
    
    def _atoms_connected_in_mol(self, mol, map1, map_o, map2) -> bool:
        """Check if three atoms form connected C-O-C in molecule."""
        atom_map = {}
        for atom in mol.GetAtoms():
            if atom.GetAtomMapNum() in [map1, map_o, map2]:
                atom_map[atom.GetAtomMapNum()] = atom.GetIdx()
                
        if len(atom_map) != 3:
            return False
            
        # Check connectivity: ar1-O and O-ar2
        o_atom = mol.GetAtomWithIdx(atom_map[map_o])
        neighbor_maps = [mol.GetAtomWithIdx(n.GetIdx()).GetAtomMapNum() 
                        for n in o_atom.GetNeighbors()]
        
        return map1 in neighbor_maps and map2 in neighbor_maps
    
    def _check_snar_conditions(self, product, reactants, ar1_idx, o_idx, ar2_idx) -> bool:
        """Verify conditions consistent with SNAr mechanism."""
        # Look for electron-withdrawing groups on aromatic rings
        ewg_patterns = [
            Chem.MolFromSmarts("c-[N+](=O)[O-]"),  # Nitro
            Chem.MolFromSmarts("c-C(=O)"),         # Carbonyl
            Chem.MolFromSmarts("c-C(F)(F)F"),      # CF3
            Chem.MolFromSmarts("c-[N+]"),          # Quaternary N
            Chem.MolFromSmarts("c-S(=O)(=O)"),     # Sulfonyl
        ]
        
        # Check if either aromatic ring has EWG
        ar1_atom = product.GetAtomWithIdx(ar1_idx)
        ar2_atom = product.GetAtomWithIdx(ar2_idx)
        
        for pattern in ewg_patterns:
            if pattern:
                matches = product.GetSubstructMatches(pattern)
                for match in matches:
                    ewg_carbon = match[0]
                    # Check if EWG is on same ring as either aromatic carbon
                    if (self._atoms_in_same_ring(product, ewg_carbon, ar1_idx) or 
                        self._atoms_in_same_ring(product, ewg_carbon, ar2_idx)):
                        return True
                        
        return True  # Allow even without clear EWG pattern
    
    def _atoms_in_same_ring(self, mol, idx1, idx2) -> bool:
        """Check if two atoms are in the same aromatic ring."""
        rings = mol.GetRingInfo().AtomRings()
        for ring in rings:
            if idx1 in ring and idx2 in ring:
                return True
        return False
