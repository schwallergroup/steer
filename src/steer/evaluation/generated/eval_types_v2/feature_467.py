"""Generated evaluation code for: Late stage aryl alkyl coupling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageArylAlkylCoupling(BaseScoring):
    """
    Checks if an aryl-alkyl C-C bond formation occurs at a specific late stage.
    Detects coupling reactions that form aryl-alkyl bonds, typically via 
    organocuprate displacement or similar cross-coupling reactions.
    """
    
    def __init__(self, config):
        self.step_position_from_end = config["parameters"].get("step_position_from_end", 1)
        # SMARTS patterns for aryl and alkyl carbons
        self.aryl_carbon = Chem.MolFromSmarts("[cH,c]")  # Aromatic carbon
        self.alkyl_carbon = Chem.MolFromSmarts("[CH3,CH2,CH]")  # Aliphatic carbon
        
    def route_scoring(self, x):
        if x < 0:
            return 0  # Bond formation doesn't happen
        else:
            # Perfect score if at exact target position, decreasing otherwise
            target_fraction = 1.0 - (self.step_position_from_end - 1) / 10.0
            return max(0, 1.0 - abs(x - target_fraction))
    
    def hit_condition(self, d):
        """Check if this reaction forms an aryl-alkyl C-C bond"""
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            reactants_smiles, product_smiles = rxn_smiles.split(">>")
            
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r) for r in reactants_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
                
            # Find bonds that exist in product but not in any reactant
            product_bonds = self._get_mapped_bonds(product)
            reactant_bonds = set()
            for reactant in reactants:
                reactant_bonds.update(self._get_mapped_bonds(reactant))
                
            new_bonds = product_bonds - reactant_bonds
            
            # Check if any new bond is an aryl-alkyl C-C bond
            for bond_atoms in new_bonds:
                if self._is_aryl_alkyl_cc_bond(product, bond_atoms):
                    return True
                    
            return False
            
        except Exception:
            return False
    
    def _get_mapped_bonds(self, mol):
        """Get set of bonds between mapped atoms as tuples of map numbers"""
        bonds = set()
        for bond in mol.GetBonds():
            atom1 = bond.GetBeginAtom()
            atom2 = bond.GetEndAtom()
            map1 = atom1.GetAtomMapNum()
            map2 = atom2.GetAtomMapNum()
            if map1 > 0 and map2 > 0:  # Both atoms are mapped
                bonds.add(tuple(sorted([map1, map2])))
        return bonds
    
    def _is_aryl_alkyl_cc_bond(self, mol, bond_atoms):
        """Check if a bond between mapped atoms is an aryl-alkyl C-C bond"""
        map1, map2 = bond_atoms
        
        # Find atoms by map numbers
        atom1 = atom2 = None
        for atom in mol.GetAtoms():
            if atom.GetAtomMapNum() == map1:
                atom1 = atom
            elif atom.GetAtomMapNum() == map2:
                atom2 = atom
                
        if not atom1 or not atom2:
            return False
            
        # Both must be carbons
        if atom1.GetSymbol() != 'C' or atom2.GetSymbol() != 'C':
            return False
            
        # Check if one is aromatic and one is aliphatic
        is_atom1_aromatic = atom1.GetIsAromatic()
        is_atom2_aromatic = atom2.GetIsAromatic()
        
        # One aromatic, one aliphatic = aryl-alkyl bond
        return (is_atom1_aromatic and not is_atom2_aromatic) or \
               (is_atom2_aromatic and not is_atom1_aromatic)
