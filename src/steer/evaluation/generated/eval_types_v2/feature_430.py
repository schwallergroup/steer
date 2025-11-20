"""Generated evaluation code for: Late stage SNAr coupling strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageSnArCoupling(BaseScoring):
    """
    Evaluates synthesis routes for late-stage nucleophilic aromatic substitution (SNAr) coupling.
    Specifically looks for SNAr reactions involving fluoride displacement in the final steps.
    """
    
    def __init__(self, config: Dict):
        self.step_position = config.get("parameters", {}).get("step_position", 1)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # SNAr reaction doesn't occur
        else:
            # Late-stage SNAr is preferred, penalize early occurrence
            return max(0, 1 - x)
    
    def hit_condition(self, d) -> bool:
        """
        Detects SNAr reactions by looking for:
        1. Aromatic fluoride in product that becomes C-N or C-O bond in reactants
        2. Presence of electron-withdrawing groups on aromatic ring
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
                
            prod_smiles, react_smiles = rxn_smiles.split(">>")
            prod_mol = Chem.MolFromSmiles(prod_smiles)
            react_mols = [Chem.MolFromSmiles(r) for r in react_smiles.split(".")]
            
            if not prod_mol or not all(react_mols):
                return False
            
            # Look for aromatic fluoride in product
            ar_fluoride_pattern = Chem.MolFromSmarts("[c:1][F:2]")
            if not prod_mol.HasSubstructMatch(ar_fluoride_pattern):
                return False
            
            # Check if fluoride is replaced by nucleophile in reactants
            fluoride_matches = prod_mol.GetSubstructMatches(ar_fluoride_pattern)
            
            for match in fluoride_matches:
                ar_carbon_map = None
                fluoride_map = None
                
                # Get atom map numbers
                for atom_idx in match:
                    atom = prod_mol.GetAtomWithIdx(atom_idx)
                    if atom.GetSymbol() == 'C':
                        ar_carbon_map = atom.GetAtomMapNum()
                    elif atom.GetSymbol() == 'F':
                        fluoride_map = atom.GetAtomMapNum()
                
                if not ar_carbon_map or not fluoride_map:
                    continue
                
                # Check if aromatic carbon is connected to nucleophile in reactants
                for react_mol in react_mols:
                    ar_carbon_atom = None
                    nucleophile_atom = None
                    
                    for atom in react_mol.GetAtoms():
                        if atom.GetAtomMapNum() == ar_carbon_map:
                            ar_carbon_atom = atom
                            break
                    
                    if not ar_carbon_atom:
                        continue
                    
                    # Check if connected to N, O, S (common nucleophiles)
                    for neighbor in ar_carbon_atom.GetNeighbors():
                        if neighbor.GetSymbol() in ['N', 'O', 'S']:
                            # Verify this isn't the same connection as in product
                            neighbor_map = neighbor.GetAtomMapNum()
                            if neighbor_map and neighbor_map != fluoride_map:
                                # Check for electron-withdrawing groups on aromatic ring
                                if self._has_electron_withdrawing_groups(react_mol, ar_carbon_atom):
                                    return True
            
            return False
            
        except Exception:
            return False
    
    def _has_electron_withdrawing_groups(self, mol, ar_carbon_atom) -> bool:
        """
        Check for common electron-withdrawing groups on the aromatic ring.
        """
        try:
            # Get the aromatic ring containing the carbon
            ring_info = mol.GetRingInfo()
            ar_rings = [ring for ring in ring_info.AtomRings() 
                       if ar_carbon_atom.GetIdx() in ring and len(ring) == 6]
            
            if not ar_rings:
                return False
            
            # Define electron-withdrawing group patterns
            ewg_patterns = [
                "[c][N+](=O)[O-]",  # Nitro
                "[c]C(=O)",         # Carbonyl
                "[c]C#N",           # Nitrile
                "[c]C(F)(F)F",      # Trifluoromethyl
                "[c]S(=O)(=O)",     # Sulfonyl
                "[c][N+]"           # Quaternary nitrogen
            ]
            
            for pattern_smarts in ewg_patterns:
                pattern = Chem.MolFromSmarts(pattern_smarts)
                if pattern and mol.HasSubstructMatch(pattern):
                    # Check if EWG is on the same ring
                    matches = mol.GetSubstructMatches(pattern)
                    for match in matches:
                        if any(atom_idx in ar_rings[0] for atom_idx in match):
                            return True
            
            return False
            
        except Exception:
            return False
