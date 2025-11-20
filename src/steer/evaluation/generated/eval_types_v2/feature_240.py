"""Generated evaluation code for: Differential halide reactivity strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class DifferentialHalideReactivity(BaseScoring):
    """
    Evaluates synthesis routes based on differential halide reactivity strategy.
    Checks if a reaction involves breaking C-Br bonds while preserving C-Cl bonds,
    exploiting the higher reactivity of bromine vs chlorine for sequential reactions.
    """
    
    def __init__(self, config: Dict):
        self.halides = config["parameters"]["halides"]  # ["Br", "Cl"]
        self.strategy = config["parameters"]["strategy"]  # "differential_reactivity"
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Strategy not used
        else:
            return 1 - x  # Earlier use of strategy is better
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction demonstrates differential halide reactivity
        by breaking C-Br bonds while preserving C-Cl bonds.
        """
        metadata = d.get("metadata", {})
        if "mapped_reaction_smiles" not in metadata:
            return False
            
        rxn_smiles = metadata["mapped_reaction_smiles"]
        rxn_parts = rxn_smiles.split(">>")
        
        if len(rxn_parts) != 2:
            return False
            
        product_smiles = rxn_parts[0]
        reactant_smiles = rxn_parts[1]
        
        try:
            product_mol = Chem.MolFromSmiles(product_smiles)
            reactant_mols = [Chem.MolFromSmiles(r) for r in reactant_smiles.split(".")]
            
            if not product_mol or not all(reactant_mols):
                return False
                
            # Check if product contains both Br and Cl
            product_has_br = self._has_halogen_bond(product_mol, "Br")
            product_has_cl = self._has_halogen_bond(product_mol, "Cl")
            
            if not (product_has_br and product_has_cl):
                return False
                
            # Check if any reactant has lost Br but retained Cl compared to product
            for reactant_mol in reactant_mols:
                reactant_has_br = self._has_halogen_bond(reactant_mol, "Br")
                reactant_has_cl = self._has_halogen_bond(reactant_mol, "Cl")
                
                # Check if this reactant demonstrates differential reactivity:
                # - Product has both Br and Cl
                # - Reactant has fewer Br atoms (some Br was consumed)
                # - Reactant still has Cl (Cl was preserved)
                if reactant_has_cl and not reactant_has_br:
                    # Check atom mapping to confirm C-Br bond was broken
                    if self._check_cbr_bond_broken(product_mol, reactant_mol):
                        return True
                        
            return False
            
        except Exception:
            return False
    
    def _has_halogen_bond(self, mol, halogen: str) -> bool:
        """Check if molecule has C-halogen bonds."""
        if not mol:
            return False
            
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == halogen:
                # Check if halogen is bonded to carbon
                for neighbor in atom.GetNeighbors():
                    if neighbor.GetSymbol() == "C":
                        return True
        return False
    
    def _check_cbr_bond_broken(self, product_mol, reactant_mol) -> bool:
        """
        Use atom mapping to verify that a C-Br bond present in product
        is broken in the reactant (Br atom is no longer connected to the same C).
        """
        if not product_mol or not reactant_mol:
            return False
            
        # Create mapping dictionaries
        product_map = {atom.GetAtomMapNum(): atom.GetIdx() 
                      for atom in product_mol.GetAtoms() 
                      if atom.GetAtomMapNum() > 0}
        reactant_map = {atom.GetAtomMapNum(): atom.GetIdx() 
                       for atom in reactant_mol.GetAtoms() 
                       if atom.GetAtomMapNum() > 0}
        
        # Find C-Br bonds in product
        for bond in product_mol.GetBonds():
            atom1 = bond.GetBeginAtom()
            atom2 = bond.GetEndAtom()
            
            # Check if this is a C-Br bond
            if ((atom1.GetSymbol() == "C" and atom2.GetSymbol() == "Br") or
                (atom1.GetSymbol() == "Br" and atom2.GetSymbol() == "C")):
                
                map1 = atom1.GetAtomMapNum()
                map2 = atom2.GetAtomMapNum()
                
                # Check if both atoms exist in reactant
                if map1 in reactant_map and map2 in reactant_map:
                    idx1 = reactant_map[map1]
                    idx2 = reactant_map[map2]
                    
                    # Check if the bond is broken in reactant
                    reactant_bond = reactant_mol.GetBondBetweenAtoms(idx1, idx2)
                    if reactant_bond is None:
                        return True
                        
        return False
