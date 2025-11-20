"""Generated evaluation code for: Late stage diaryl ether formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageArylEtherFormation(BaseScoring):
    """
    Evaluates synthesis routes for late-stage diaryl ether formation.
    
    This class identifies reactions that form diaryl ether bonds (Ar-O-Ar) 
    and rewards routes where this key bond formation occurs late in the synthesis,
    typically via coupling reactions like Buchwald-Hartwig etherification.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "float")
        self.target_depth = config.get("target_depth", {}).get("value", 0.8)
    
    def route_scoring(self, x) -> float:
        """
        Convert depth fraction to score (0-10 scale).
        Late-stage formation (higher x values) gets higher scores.
        """
        if x < 0:
            return 0  # Diaryl ether formation doesn't occur
        else:
            # Late-stage formation is better - score increases with depth
            return x * 10
    
    def hit_condition(self, d):
        """
        Check if this reaction forms a diaryl ether bond.
        Returns True if an aromatic C-O-aromatic bond is formed.
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            reactants_smiles, product_smiles = rxn_smiles.split(">>")
            
            # Parse molecules
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Find diaryl ether patterns in product
            diaryl_ether_pattern = Chem.MolFromSmarts("[c,C:1]-O-[c,C:2]")
            if not product.HasSubstructMatch(diaryl_ether_pattern):
                return False
            
            # Check if this ether bond is newly formed
            ether_matches = product.GetSubstructMatches(diaryl_ether_pattern)
            
            for match in ether_matches:
                ar1_map = None
                ar2_map = None
                
                # Get atom map numbers for the aromatic carbons
                for atom in product.GetAtoms():
                    if atom.GetIdx() == match[0]:  # First aromatic carbon
                        ar1_map = atom.GetAtomMapNum()
                    elif atom.GetIdx() == match[1]:  # Second aromatic carbon
                        ar2_map = atom.GetAtomMapNum()
                
                if not ar1_map or not ar2_map:
                    continue
                
                # Check if these atoms are in different reactant molecules
                ar1_reactant = None
                ar2_reactant = None
                
                for i, reactant in enumerate(reactants):
                    reactant_maps = [atom.GetAtomMapNum() for atom in reactant.GetAtoms()]
                    if ar1_map in reactant_maps:
                        ar1_reactant = i
                    if ar2_map in reactant_maps:
                        ar2_reactant = i
                
                # If the aromatic carbons are from different reactants, 
                # this indicates ether bond formation
                if (ar1_reactant is not None and ar2_reactant is not None and 
                    ar1_reactant != ar2_reactant):
                    
                    # Additional check: ensure at least one reactant has aromatic ring
                    ar1_aromatic = self._is_carbon_aromatic_in_reactant(reactants[ar1_reactant], ar1_map)
                    ar2_aromatic = self._is_carbon_aromatic_in_reactant(reactants[ar2_reactant], ar2_map)
                    
                    if ar1_aromatic and ar2_aromatic:
                        return True
            
            return False
            
        except Exception:
            return False
    
    def _is_carbon_aromatic_in_reactant(self, reactant_mol, atom_map_num):
        """Helper method to check if a mapped atom is aromatic in the reactant."""
        for atom in reactant_mol.GetAtoms():
            if atom.GetAtomMapNum() == atom_map_num:
                return atom.GetIsAromatic()
        return False
