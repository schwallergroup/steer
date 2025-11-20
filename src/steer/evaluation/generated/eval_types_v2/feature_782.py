"""Generated evaluation code for: Early stage Sandmeyer bromination"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlyStageSandmeyerBromination(BaseScoring):
    """
    Evaluates if a Sandmeyer bromination reaction (C-N to C-Br conversion) occurs early in the synthesis.
    Returns higher scores for earlier occurrence of the reaction.
    """
    
    def __init__(self, config: Dict):
        self.timing_preference = config.get("timing", "early")
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Sandmeyer reaction doesn't occur
        else:
            # Early stage is preferred, so lower depth fraction = higher score
            return 1 - x
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction represents a Sandmeyer bromination (C-N to C-Br)"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
        
        try:
            reactants_smiles, products_smiles = mapped_rxn.split(">>")
            product_mol = Chem.MolFromSmiles(reactants_smiles)  # Note: reversed because it's retrosynthetic
            reactant_mols = [Chem.MolFromSmiles(smi) for smi in products_smiles.split(".")]
            
            if not product_mol or not all(reactant_mols):
                return False
            
            # Find aromatic carbons bonded to nitrogen in product
            product_cn_pairs = self._find_aromatic_cn_bonds(product_mol)
            
            # Find aromatic carbons bonded to bromine in reactants
            reactant_cbr_pairs = []
            for mol in reactant_mols:
                reactant_cbr_pairs.extend(self._find_aromatic_cbr_bonds(mol))
            
            # Check if any C-N bond in product corresponds to C-Br bond in reactants
            # by matching atom map numbers
            for prod_c_map, prod_n_map in product_cn_pairs:
                for react_c_map, react_br_map in reactant_cbr_pairs:
                    if prod_c_map == react_c_map:
                        # Same carbon atom, check if nitrogen was replaced by bromine
                        return True
            
            return False
            
        except Exception:
            return False
    
    def _find_aromatic_cn_bonds(self, mol):
        """Find aromatic carbon-nitrogen bonds and return atom map numbers"""
        cn_pairs = []
        for bond in mol.GetBonds():
            atom1 = bond.GetBeginAtom()
            atom2 = bond.GetEndAtom()
            
            # Check for aromatic C-N bond
            if ((atom1.GetSymbol() == 'C' and atom2.GetSymbol() == 'N') or
                (atom1.GetSymbol() == 'N' and atom2.GetSymbol() == 'C')):
                
                c_atom = atom1 if atom1.GetSymbol() == 'C' else atom2
                n_atom = atom2 if atom1.GetSymbol() == 'C' else atom1
                
                # Check if carbon is aromatic
                if c_atom.GetIsAromatic():
                    c_map = c_atom.GetAtomMapNum()
                    n_map = n_atom.GetAtomMapNum()
                    if c_map > 0 and n_map > 0:  # Only consider mapped atoms
                        cn_pairs.append((c_map, n_map))
        
        return cn_pairs
    
    def _find_aromatic_cbr_bonds(self, mol):
        """Find aromatic carbon-bromine bonds and return atom map numbers"""
        cbr_pairs = []
        for bond in mol.GetBonds():
            atom1 = bond.GetBeginAtom()
            atom2 = bond.GetEndAtom()
            
            # Check for aromatic C-Br bond
            if ((atom1.GetSymbol() == 'C' and atom2.GetSymbol() == 'Br') or
                (atom1.GetSymbol() == 'Br' and atom2.GetSymbol() == 'C')):
                
                c_atom = atom1 if atom1.GetSymbol() == 'C' else atom2
                br_atom = atom2 if atom1.GetSymbol() == 'C' else atom1
                
                # Check if carbon is aromatic
                if c_atom.GetIsAromatic():
                    c_map = c_atom.GetAtomMapNum()
                    br_map = br_atom.GetAtomMapNum()
                    if c_map > 0 and br_map > 0:  # Only consider mapped atoms
                        cbr_pairs.append((c_map, br_map))
        
        return cbr_pairs
