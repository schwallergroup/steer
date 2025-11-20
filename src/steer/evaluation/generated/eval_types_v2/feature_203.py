"""Generated evaluation code for: Late stage SNAr fluorine substitution"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageArylFluorideDisconnection(BaseScoring):
    """
    Evaluates routes for late-stage SNAr fluorine substitution reactions.
    Detects aromatic C-F bond breaking in electron-deficient systems.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "float")
        self.target_depth = config.get("target_depth", {}).get("value", 0.8)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # SNAr disconnection doesn't happen
        else:
            return 1 - x  # Later disconnection is better for late-stage
    
    def hit_condition(self, d) -> bool:
        """Check if reaction involves SNAr fluorine substitution"""
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            rxn_parts = rxn_smiles.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            product = Chem.MolFromSmiles(rxn_parts[0])
            reactants = [Chem.MolFromSmiles(r) for r in rxn_parts[1].split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check for C-F bond breaking (F in product but not in one reactant)
            if not self._has_aromatic_fluorine(product):
                return False
            
            # Find fluorine atoms with map numbers in product
            product_f_maps = set()
            for atom in product.GetAtoms():
                if atom.GetSymbol() == 'F' and atom.GetIsAromatic() and atom.GetAtomMapNum() > 0:
                    product_f_maps.add(atom.GetAtomMapNum())
            
            if not product_f_maps:
                return False
            
            # Check if any fluorine is missing in reactants (indicating substitution)
            for f_map in product_f_maps:
                found_in_reactants = False
                for reactant in reactants:
                    for atom in reactant.GetAtoms():
                        if atom.GetAtomMapNum() == f_map:
                            found_in_reactants = True
                            break
                    if found_in_reactants:
                        break
                
                # If fluorine not found in reactants, it's a substitution
                if not found_in_reactants:
                    # Verify it's on electron-deficient aromatic system
                    if self._is_electron_deficient_aromatic(product, f_map):
                        return True
            
            return False
            
        except Exception:
            return False
    
    def _has_aromatic_fluorine(self, mol) -> bool:
        """Check if molecule has aromatic fluorine"""
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'F' and atom.GetIsAromatic():
                return True
        return False
    
    def _is_electron_deficient_aromatic(self, mol, f_map_num) -> bool:
        """Check if fluorine is on electron-deficient aromatic system"""
        # Find the fluorine atom and its aromatic ring
        f_atom = None
        for atom in mol.GetAtoms():
            if atom.GetAtomMapNum() == f_map_num:
                f_atom = atom
                break
        
        if not f_atom or not f_atom.GetIsAromatic():
            return False
        
        # Get the aromatic ring containing the fluorine
        ring_info = mol.GetRingInfo()
        for ring in ring_info.AtomRings():
            if f_atom.GetIdx() in ring:
                # Check for electron-withdrawing groups or heteroatoms
                for atom_idx in ring:
                    atom = mol.GetAtomByIdx(atom_idx)
                    # Heteroatoms (N, O, S) make ring electron-deficient
                    if atom.GetSymbol() in ['N', 'O', 'S']:
                        return True
                    
                    # Check for electron-withdrawing substituents
                    for neighbor in atom.GetNeighbors():
                        if neighbor.GetIdx() not in ring:
                            # Common EWGs: NO2, CN, carbonyl, etc.
                            if neighbor.GetSymbol() == 'N':
                                # Check for NO2
                                o_count = sum(1 for n in neighbor.GetNeighbors() 
                                            if n.GetSymbol() == 'O')
                                if o_count >= 2:
                                    return True
                            elif neighbor.GetSymbol() == 'C':
                                # Check for CN or carbonyl
                                if any(n.GetSymbol() == 'N' for n in neighbor.GetNeighbors()):
                                    return True
                                if any(n.GetSymbol() == 'O' and mol.GetBondBetweenAtoms(
                                    neighbor.GetIdx(), n.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
                                    for n in neighbor.GetNeighbors()):
                                    return True
        
        return False
