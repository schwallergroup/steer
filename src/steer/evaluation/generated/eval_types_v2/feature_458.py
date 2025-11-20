"""Generated evaluation code for: Sandmeyer reaction for dihalide installation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SandmeyerReaction(BaseScoring):
    """
    Evaluates synthesis routes for the presence of Sandmeyer reactions that convert 
    amino groups to halides, particularly for dihalide installation on pyridine substrates.
    
    The Sandmeyer reaction involves conversion of an aromatic amine (via diazonium salt) 
    to a halide, which is important for creating dibromo intermediates for SNAr reactions.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "bool")
        self.target_depth = config.get("target_depth", {}).get("value", 0)
    
    def route_scoring(self, x) -> float:
        if self.condition_type == "bool":
            if x < 0:
                return 0  # Sandmeyer reaction not found
            else:
                return 1  # Sandmeyer reaction found
        else:
            if x < 0:
                return 0  # Sandmeyer reaction not found
            # Earlier Sandmeyer reaction is generally better for synthetic efficiency
            return 1 - x if x <= 1.0 else 0
    
    def hit_condition(self, d):
        """
        Detects Sandmeyer reaction by checking for:
        1. Loss of amino group from aromatic system
        2. Gain of halide (Br, Cl, I) at same position
        3. Presence of pyridine or similar aromatic substrate
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            products_smiles, reactants_smiles = rxn_smiles.split(">>")
            
            # Parse molecules
            products = [Chem.MolFromSmiles(smi) for smi in products_smiles.split(".")]
            reactants = [Chem.MolFromSmiles(smi) for smi in reactants_smiles.split(".")]
            
            if not all(products) or not all(reactants):
                return False
            
            # Check for amino group loss and halide gain
            for prod in products:
                for react in reactants:
                    if self._is_sandmeyer_transformation(prod, react):
                        return True
            
            return False
            
        except Exception:
            return False
    
    def _is_sandmeyer_transformation(self, product_mol, reactant_mol):
        """
        Check if transformation represents amino -> halide conversion on aromatic ring
        """
        # Define patterns for aromatic amines (including pyridine)
        aromatic_amine_patterns = [
            "[c,n:1][NH2:2]",  # Aromatic amine
            "[c:1][NH2:2]",    # Aromatic carbon with amine
            "c1cc[c,n]([NH2])cc1"  # Pyridine/benzene amine pattern
        ]
        
        # Define patterns for aromatic halides
        aromatic_halide_patterns = [
            "[c,n:1][Br:2]",   # Aromatic bromide
            "[c,n:1][Cl:2]",   # Aromatic chloride
            "[c,n:1][I:2]",    # Aromatic iodide
        ]
        
        # Check if reactant has aromatic amine
        has_aromatic_amine = False
        for pattern in aromatic_amine_patterns:
            if reactant_mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                has_aromatic_amine = True
                break
        
        if not has_aromatic_amine:
            return False
        
        # Check if product has aromatic halide
        has_aromatic_halide = False
        for pattern in aromatic_halide_patterns:
            if product_mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                has_aromatic_halide = True
                break
        
        if not has_aromatic_halide:
            return False
        
        # Additional check: ensure the transformation involves pyridine substrate
        pyridine_pattern = Chem.MolFromSmarts("c1ccncc1")
        has_pyridine_substrate = (reactant_mol.HasSubstructMatch(pyridine_pattern) or 
                                 product_mol.HasSubstructMatch(pyridine_pattern))
        
        # Verify atom mapping consistency if available
        if has_aromatic_amine and has_aromatic_halide and has_pyridine_substrate:
            return self._check_atom_mapping_consistency(product_mol, reactant_mol)
        
        return False
    
    def _check_atom_mapping_consistency(self, product_mol, reactant_mol):
        """
        Verify that the amino group position in reactant corresponds to halide position in product
        """
        try:
            # Get atom mappings
            reactant_map = {atom.GetAtomMapNum(): atom.GetIdx() 
                           for atom in reactant_mol.GetAtoms() if atom.GetAtomMapNum() > 0}
            product_map = {atom.GetAtomMapNum(): atom.GetIdx() 
                          for atom in product_mol.GetAtoms() if atom.GetAtomMapNum() > 0}
            
            if not reactant_map or not product_map:
                return True  # No mapping available, assume valid
            
            # Find common mapped positions and check transformation
            common_maps = set(reactant_map.keys()) & set(product_map.keys())
            
            for map_num in common_maps:
                r_idx = reactant_map[map_num]
                p_idx = product_map[map_num]
                
                r_atom = reactant_mol.GetAtomWithIdx(r_idx)
                p_atom = product_mol.GetAtomWithIdx(p_idx)
                
                # Check if this position shows amino -> halide conversion
                r_neighbors = [n.GetSymbol() for n in r_atom.GetNeighbors()]
                p_neighbors = [n.GetSymbol() for n in p_atom.GetNeighbors()]
                
                if 'N' in r_neighbors and any(hal in p_neighbors for hal in ['Br', 'Cl', 'I']):
                    return True
            
            return False
            
        except Exception:
            return True  # Default to True if mapping check fails
