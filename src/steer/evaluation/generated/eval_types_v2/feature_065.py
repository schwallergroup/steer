"""Generated evaluation code for: Late stage C-S cross-coupling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageCsCrossCoupling(BaseScoring):
    """
    Evaluates synthesis routes for late-stage C-S cross-coupling reactions.
    Rewards routes where a C-S bond formation occurs in the final step,
    typically via palladium-catalyzed cross-coupling to form diaryl thioethers.
    """
    
    def __init__(self, config: Dict):
        self.step_position = config.get("step_position", "final")
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # C-S coupling doesn't occur
        elif x == 0:
            return 10  # Perfect - occurs in final step
        else:
            # Penalize earlier occurrence, but still reward presence
            return max(0, 8 - x * 2)
    
    def hit_condition(self, d) -> bool:
        """
        Detects C-S cross-coupling by checking for:
        1. Formation of C-S bond between aromatic carbons and sulfur
        2. Typical cross-coupling patterns (aryl halide + thiol/thiolate)
        """
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            product_smiles, reactants_smiles = rxn_smiles.split(">>")
            
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r) for r in reactants_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check if C-S bond is formed in product
            cs_bonds_product = self._find_aromatic_cs_bonds(product)
            if not cs_bonds_product:
                return False
            
            # Check if the C-S bond atoms are separated in reactants
            for cs_bond in cs_bonds_product:
                c_mapnum, s_mapnum = cs_bond
                if self._atoms_separated_in_reactants(reactants, c_mapnum, s_mapnum):
                    # Additional check for cross-coupling patterns
                    if self._has_cross_coupling_pattern(reactants, c_mapnum, s_mapnum):
                        return True
            
            return False
            
        except Exception:
            return False
    
    def _find_aromatic_cs_bonds(self, mol):
        """Find C-S bonds where carbon is aromatic"""
        cs_bonds = []
        for bond in mol.GetBonds():
            atom1 = bond.GetBeginAtom()
            atom2 = bond.GetEndAtom()
            
            # Check for aromatic C-S bond
            if ((atom1.GetSymbol() == 'C' and atom1.GetIsAromatic() and atom2.GetSymbol() == 'S') or
                (atom2.GetSymbol() == 'C' and atom2.GetIsAromatic() and atom1.GetSymbol() == 'S')):
                
                map1 = atom1.GetAtomMapNum()
                map2 = atom2.GetAtomMapNum()
                if map1 > 0 and map2 > 0:
                    if atom1.GetSymbol() == 'C':
                        cs_bonds.append((map1, map2))
                    else:
                        cs_bonds.append((map2, map1))
        
        return cs_bonds
    
    def _atoms_separated_in_reactants(self, reactants, c_mapnum, s_mapnum):
        """Check if carbon and sulfur atoms are in different reactant molecules"""
        c_mol = None
        s_mol = None
        
        for mol in reactants:
            has_c = any(atom.GetAtomMapNum() == c_mapnum for atom in mol.GetAtoms())
            has_s = any(atom.GetAtomMapNum() == s_mapnum for atom in mol.GetAtoms())
            
            if has_c:
                c_mol = mol
            if has_s:
                s_mol = mol
        
        return c_mol is not None and s_mol is not None and c_mol != s_mol
    
    def _has_cross_coupling_pattern(self, reactants, c_mapnum, s_mapnum):
        """Check for typical cross-coupling patterns: aryl halide + thiol/thiolate"""
        c_mol = None
        s_mol = None
        
        # Find molecules containing the mapped atoms
        for mol in reactants:
            if any(atom.GetAtomMapNum() == c_mapnum for atom in mol.GetAtoms()):
                c_mol = mol
            if any(atom.GetAtomMapNum() == s_mapnum for atom in mol.GetAtoms()):
                s_mol = mol
        
        if not c_mol or not s_mol:
            return False
        
        # Check for aryl halide pattern (aromatic C with halogen neighbor)
        has_aryl_halide = False
        for atom in c_mol.GetAtoms():
            if atom.GetAtomMapNum() == c_mapnum:
                for neighbor in atom.GetNeighbors():
                    if neighbor.GetSymbol() in ['Cl', 'Br', 'I']:
                        has_aryl_halide = True
                        break
        
        # Check for thiol/thiolate pattern
        has_thiol_pattern = False
        thiol_patterns = [
            Chem.MolFromSmarts('[SH1]'),  # thiol
            Chem.MolFromSmarts('[S-]'),   # thiolate
            Chem.MolFromSmarts('S')       # general sulfur
        ]
        
        for pattern in thiol_patterns:
            if pattern and s_mol.HasSubstructMatch(pattern):
                has_thiol_pattern = True
                break
        
        return has_aryl_halide and has_thiol_pattern
