"""Generated evaluation code for: Late stage tertiary alcohol deoxygenation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageDeoxygentaion(BaseScoring):
    """
    Evaluates synthesis routes for late-stage tertiary alcohol deoxygenation reactions.
    Rewards routes where tertiary benzylic alcohols are deoxygenated in later stages.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "depth")
        self.target_depth = config.get("target_depth", {}).get("value", 0.8)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Reaction doesn't occur
        else:
            # Higher score for later stage reactions (higher depth fraction)
            # Scale to 0-10 range with preference for depth > 0.7
            if x >= 0.7:
                return 8 + 2 * (x - 0.7) / 0.3  # 8-10 for depth 0.7-1.0
            else:
                return 8 * x / 0.7  # 0-8 for depth 0-0.7
    
    def hit_condition(self, d):
        """Check if reaction involves tertiary benzylic alcohol deoxygenation"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants = rxn_parts[0]
            products = rxn_parts[1].split(".")
            
            # Parse reactant molecule
            reactant_mol = Chem.MolFromSmiles(reactants)
            if not reactant_mol:
                return False
            
            # Check for tertiary benzylic alcohol in reactant
            if not self._has_tertiary_benzylic_alcohol(reactant_mol):
                return False
            
            # Check if products show deoxygenation (loss of OH group)
            product_mols = [Chem.MolFromSmiles(p) for p in products if Chem.MolFromSmiles(p)]
            
            # Find mapped atoms with OH groups in reactant
            oh_carbons = self._find_tertiary_benzylic_oh_carbons(reactant_mol)
            
            # Check if any OH carbon is deoxygenated in products
            for carbon_map in oh_carbons:
                if self._is_carbon_deoxygenated(carbon_map, reactant_mol, product_mols):
                    return True
                    
            return False
            
        except Exception:
            return False
    
    def _has_tertiary_benzylic_alcohol(self, mol):
        """Check if molecule has tertiary benzylic alcohol"""
        # SMARTS for tertiary carbon with OH adjacent to aromatic ring
        tertiary_benzylic_oh_pattern = "[OH1][C;X4;H0](c)([!H])([!H])"
        pattern = Chem.MolFromSmarts(tertiary_benzylic_oh_pattern)
        return mol.HasSubstructMatch(pattern) if pattern else False
    
    def _find_tertiary_benzylic_oh_carbons(self, mol):
        """Find atom map numbers of tertiary benzylic carbons with OH groups"""
        tertiary_benzylic_oh_pattern = "[OH1][C;X4;H0](c)([!H])([!H])"
        pattern = Chem.MolFromSmarts(tertiary_benzylic_oh_pattern)
        oh_carbons = []
        
        if pattern:
            matches = mol.GetSubstructMatches(pattern)
            for match in matches:
                carbon_idx = match[1]  # Second atom in pattern is the carbon
                carbon_atom = mol.GetAtomWithIdx(carbon_idx)
                map_num = carbon_atom.GetAtomMapNum()
                if map_num > 0:
                    oh_carbons.append(map_num)
                    
        return oh_carbons
    
    def _is_carbon_deoxygenated(self, carbon_map, reactant_mol, product_mols):
        """Check if mapped carbon loses OH group in products"""
        # Find carbon in reactant
        reactant_carbon = None
        for atom in reactant_mol.GetAtoms():
            if atom.GetAtomMapNum() == carbon_map:
                reactant_carbon = atom
                break
        
        if not reactant_carbon:
            return False
            
        # Count OH groups attached to this carbon in reactant
        reactant_oh_count = sum(1 for neighbor in reactant_carbon.GetNeighbors() 
                              if neighbor.GetSymbol() == 'O' and neighbor.GetTotalNumHs() > 0)
        
        # Find same carbon in products and count OH groups
        for product_mol in product_mols:
            for atom in product_mol.GetAtoms():
                if atom.GetAtomMapNum() == carbon_map:
                    product_oh_count = sum(1 for neighbor in atom.GetNeighbors() 
                                         if neighbor.GetSymbol() == 'O' and neighbor.GetTotalNumHs() > 0)
                    # Deoxygenation occurred if OH count decreased
                    if reactant_oh_count > product_oh_count:
                        return True
                        
        return False
