"""Generated evaluation code for: Convergent synthesis via two fragments"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentStrategy(BaseScoring):
    """
    Evaluates convergent synthesis strategy by checking if the route assembles 
    multiple fragments at a specific depth through coupling reactions.
    """
    
    def __init__(self, config: Dict):
        self.target_fragment_count = config["fragment_count"]
        self.target_coupling_depth = config["coupling_reaction_depth"]
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Convergent coupling doesn't happen
        else:
            # Prefer coupling at target depth, penalize deviation
            depth_penalty = abs(x - (self.target_coupling_depth / 10.0))
            return max(0, 1 - depth_penalty)
    
    def hit_condition(self, d):
        """Check if this reaction represents a convergent coupling of fragments"""
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            rxn_parts = rxn_smiles.split(">>")
            
            if len(rxn_parts) != 2:
                return False
                
            product = rxn_parts[0]
            reactants = rxn_parts[1].split(".")
            
            # Check if we have the target number of reactant fragments
            if len(reactants) != self.target_fragment_count:
                return False
            
            # Parse molecules
            prod_mol = Chem.MolFromSmiles(product)
            reactant_mols = [Chem.MolFromSmiles(r) for r in reactants]
            
            if not prod_mol or not all(reactant_mols):
                return False
            
            # Check if this is a coupling reaction (bond formation between fragments)
            return self._is_coupling_reaction(prod_mol, reactant_mols)
            
        except (KeyError, AttributeError, ValueError):
            return False
    
    def _is_coupling_reaction(self, product, reactants):
        """
        Determine if the reaction represents coupling by checking:
        1. Product has more bonds than sum of reactants
        2. Key coupling functional groups are involved
        """
        # Count bonds in product vs reactants
        prod_bond_count = product.GetNumBonds()
        reactant_bond_count = sum(mol.GetNumBonds() for mol in reactants)
        
        # Must form new bonds (accounting for potential bond breaking/forming)
        if prod_bond_count <= reactant_bond_count:
            return False
        
        # Check for common coupling reaction patterns
        coupling_patterns = [
            # Amide formation
            "[C](=[O])[N]",  # Amide bond
            # Ester formation  
            "[C](=[O])[O][C]",  # Ester bond
            # C-C coupling patterns
            "[c,C][c,C]",  # Aromatic or aliphatic C-C
            # Suzuki-like patterns
            "[c][c]",  # Aromatic C-C coupling
        ]
        
        for pattern in coupling_patterns:
            pattern_mol = Chem.MolFromSmarts(pattern)
            if pattern_mol and product.HasSubstructMatch(pattern_mol):
                # Check that this pattern spans across original fragments
                if self._pattern_spans_fragments(product, reactants, pattern_mol):
                    return True
        
        return False
    
    def _pattern_spans_fragments(self, product, reactants, pattern_mol):
        """
        Check if the matched pattern represents a bond between original fragments
        by using atom mapping numbers to trace fragment origins
        """
        matches = product.GetSubstructMatches(pattern_mol)
        
        for match in matches:
            # Get atom map numbers for the matched atoms
            map_nums = []
            for atom_idx in match:
                atom = product.GetAtomWithIdx(atom_idx)
                map_num = atom.GetAtomMapNum()
                if map_num > 0:
                    map_nums.append(map_num)
            
            if len(map_nums) >= 2:
                # Check if these atoms come from different reactant fragments
                fragment_origins = []
                for map_num in map_nums:
                    for i, reactant in enumerate(reactants):
                        if any(a.GetAtomMapNum() == map_num for a in reactant.GetAtoms()):
                            fragment_origins.append(i)
                            break
                
                # If atoms come from different fragments, it's a coupling bond
                if len(set(fragment_origins)) > 1:
                    return True
        
        return False
