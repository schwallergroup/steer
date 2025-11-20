"""Generated evaluation code for: Halodesilylation for regioselective halogen introduction"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class HalodesilylationDetection(BaseScoring):
    """
    Detects halodesilylation reactions where a trimethylsilyl (TMS) group is replaced with a halogen.
    This reaction type is used for regioselective halogen introduction.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "bool")
        self.target_depth = config.get("target_depth", {}).get("value", -1)
    
    def route_scoring(self, x) -> float:
        """Convert depth fraction to score (0-10 scale)"""
        if self.condition_type == "bool":
            if self.target_depth == -1:  # Positive if condition met
                return 1 if x >= 0 else 0
        else:
            if x < 0:
                return 0  # Reaction not found
            return max(0, 1 - abs(x - self.target_depth))
    
    def hit_condition(self, d) -> bool:
        """Check if a reaction node represents halodesilylation"""
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
                
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse molecules
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            if not all(reactants) or not all(products):
                return False
            
            # Check for TMS group in reactants and halogen in products at same position
            return self._detect_tms_to_halogen_transformation(reactants, products)
            
        except Exception:
            return False
    
    def _detect_tms_to_halogen_transformation(self, reactants, products) -> bool:
        """Detect TMS to halogen transformation using atom mapping"""
        # TMS group pattern: Si connected to three methyls
        tms_pattern = Chem.MolFromSmarts("[Si]([CH3])([CH3])[CH3]")
        if not tms_pattern:
            return False
        
        # Find TMS groups in reactants
        for reactant in reactants:
            if reactant.HasSubstructMatch(tms_pattern):
                # Get atom map numbers for atoms connected to Si
                tms_matches = reactant.GetSubstructMatches(tms_pattern)
                
                for match in tms_matches:
                    si_idx = match[0]  # Silicon atom index
                    si_atom = reactant.GetAtomWithIdx(si_idx)
                    
                    # Find carbon atom connected to Si (not the methyl groups)
                    for neighbor in si_atom.GetNeighbors():
                        if neighbor.GetSymbol() == 'C':
                            # Check if this carbon has other non-methyl substituents
                            non_methyl_neighbors = []
                            for c_neighbor in neighbor.GetNeighbors():
                                if c_neighbor.GetIdx() != si_idx:
                                    non_methyl_neighbors.append(c_neighbor)
                            
                            if len(non_methyl_neighbors) > 0:  # Carbon connected to rest of molecule
                                map_num = neighbor.GetAtomMapNum()
                                if map_num > 0:
                                    # Check if this position has halogen in products
                                    if self._has_halogen_at_position(products, map_num):
                                        return True
        
        return False
    
    def _has_halogen_at_position(self, products, map_num) -> bool:
        """Check if a halogen exists at the specified atom map position in products"""
        halogens = {'F', 'Cl', 'Br', 'I'}
        
        for product in products:
            for atom in product.GetAtoms():
                if atom.GetAtomMapNum() == map_num:
                    # Check if this atom or its neighbors are halogens
                    if atom.GetSymbol() in halogens:
                        return True
                    # Check neighbors for newly formed C-X bonds
                    for neighbor in atom.GetNeighbors():
                        if neighbor.GetSymbol() in halogens:
                            return True
        
        return False
