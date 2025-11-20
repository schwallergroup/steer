"""Generated evaluation code for: Late stage ester formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageEsterFormation(BaseScoring):
    """
    Evaluates whether ester formation occurs late in the synthesis route.
    Returns higher scores when ester formation happens after the stage threshold.
    """
    
    def __init__(self, config: Dict):
        self.stage_threshold = config.get("stage_threshold", 0.7)
        # SMARTS pattern for ester formation - looking for C(=O)O-C bond formation
        self.ester_pattern = Chem.MolFromSmarts("[C:1](=[O:2])[O:3][C:4]")
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ester formation doesn't occur
        
        if x >= self.stage_threshold:
            return 10  # Perfect score for late-stage ester formation
        else:
            # Linear scaling: earlier reactions get lower scores
            return (x / self.stage_threshold) * 10
    
    def hit_condition(self, d) -> bool:
        """
        Check if the reaction involves ester formation by detecting:
        1. Formation of ester bond (C(=O)O-C pattern appears in product but not all reactants)
        2. Common esterification reaction patterns
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
        
        try:
            reactants_smiles, product_smiles = mapped_rxn.split(">>")
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r) for r in reactants_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check if ester pattern is present in product
            if not product.HasSubstructMatch(self.ester_pattern):
                return False
            
            # Get ester matches in product
            product_ester_atoms = set()
            for match in product.GetSubstructMatches(self.ester_pattern):
                product_ester_atoms.update(self._get_mapped_atoms(product, match))
            
            # Check if this ester bond is newly formed (not present in all reactants)
            for reactant in reactants:
                if reactant.HasSubstructMatch(self.ester_pattern):
                    for match in reactant.GetSubstructMatches(self.ester_pattern):
                        reactant_ester_atoms = set(self._get_mapped_atoms(reactant, match))
                        # If the same ester bond exists in reactant, remove from consideration
                        product_ester_atoms -= reactant_ester_atoms
            
            # If we have remaining ester atoms, it means new ester formation occurred
            return len(product_ester_atoms) > 0
            
        except Exception:
            return False
    
    def _get_mapped_atoms(self, mol, match_indices):
        """Helper method to get atom map numbers for matched atoms"""
        mapped_atoms = []
        for idx in match_indices:
            atom = mol.GetAtomWithIdx(idx)
            map_num = atom.GetAtomMapNum()
            if map_num > 0:
                mapped_atoms.append(map_num)
        return mapped_atoms
