"""Generated evaluation code for: Selective terminal alkene hydrogenation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SelectiveTerminalAlkeneHydrogenation(BaseScoring):
    """
    Checks if a route employs selective terminal alkene hydrogenation while preserving internal alkenes.
    Detects reactions that reduce terminal C=C bonds in molecules that also contain internal C=C bonds.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "fraction")
        self.target_depth = config.get("target_depth", {}).get("value", 0.0)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Condition not met
        else:
            if self.condition_type == "bool":
                return 1  # Condition met
            else:
                return 1 - x  # Earlier is better (closer to target)
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction performs selective terminal alkene hydrogenation"""
        try:
            rxn = d["metadata"]["mapped_reaction_smiles"].split(">>")
            reactants = [Chem.MolFromSmiles(r) for r in rxn[0].split(".")]
            products = [Chem.MolFromSmiles(p) for p in rxn[1].split(".")]
            
            # Check each reactant-product pair for selective hydrogenation
            for reactant in reactants:
                if reactant is None:
                    continue
                    
                # Check if reactant has both terminal and internal alkenes
                if not self._has_terminal_and_internal_alkenes(reactant):
                    continue
                
                # Find corresponding product (same carbon skeleton)
                for product in products:
                    if product is None:
                        continue
                    
                    if self._is_selective_terminal_hydrogenation(reactant, product):
                        return True
                        
            return False
            
        except Exception:
            return False
    
    def _has_terminal_and_internal_alkenes(self, mol) -> bool:
        """Check if molecule has both terminal and internal C=C bonds"""
        terminal_alkene = Chem.MolFromSmarts("[CH2]=[CH2]")  # Terminal alkene
        internal_alkene = Chem.MolFromSmarts("[CH,C]=[CH,C]")  # Internal alkene (at least one non-H substituent)
        
        has_terminal = mol.HasSubstructMatch(terminal_alkene)
        
        # Check for internal alkenes (exclude terminal matches)
        internal_matches = mol.GetSubstructMatches(internal_alkene)
        has_internal = False
        
        for match in internal_matches:
            atom1, atom2 = match
            # Check if this is truly internal (not terminal)
            if not self._is_terminal_alkene_atoms(mol, atom1, atom2):
                has_internal = True
                break
        
        return has_terminal and has_internal
    
    def _is_terminal_alkene_atoms(self, mol, atom1_idx, atom2_idx) -> bool:
        """Check if the C=C bond between two atoms is terminal"""
        atom1 = mol.GetAtomWithIdx(atom1_idx)
        atom2 = mol.GetAtomWithIdx(atom2_idx)
        
        # Count carbon neighbors for each atom in the double bond
        carbon_neighbors_1 = sum(1 for neighbor in atom1.GetNeighbors() 
                                if neighbor.GetAtomicNum() == 6 and neighbor.GetIdx() != atom2_idx)
        carbon_neighbors_2 = sum(1 for neighbor in atom2.GetNeighbors() 
                                if neighbor.GetAtomicNum() == 6 and neighbor.GetIdx() != atom1_idx)
        
        # Terminal alkene: one carbon has no other carbon neighbors
        return carbon_neighbors_1 == 0 or carbon_neighbors_2 == 0
    
    def _is_selective_terminal_hydrogenation(self, reactant, product) -> bool:
        """Check if product is result of selective terminal alkene hydrogenation"""
        # Count terminal and internal alkenes in reactant and product
        terminal_pattern = Chem.MolFromSmarts("[CH2]=[CH2]")
        
        reactant_terminal = len(reactant.GetSubstructMatches(terminal_pattern))
        product_terminal = len(product.GetSubstructMatches(terminal_pattern))
        
        # Check if terminal alkene count decreased
        terminal_reduced = reactant_terminal > product_terminal
        
        if not terminal_reduced:
            return False
        
        # Check if internal alkenes are preserved
        reactant_internal_count = self._count_internal_alkenes(reactant)
        product_internal_count = self._count_internal_alkenes(product)
        
        # Internal alkenes should be preserved (same count)
        internal_preserved = reactant_internal_count == product_internal_count
        
        return terminal_reduced and internal_preserved
    
    def _count_internal_alkenes(self, mol) -> int:
        """Count internal C=C bonds in molecule"""
        alkene_pattern = Chem.MolFromSmarts("[CH,C]=[CH,C]")
        matches = mol.GetSubstructMatches(alkene_pattern)
        
        internal_count = 0
        for match in matches:
            atom1_idx, atom2_idx = match
            if not self._is_terminal_alkene_atoms(mol, atom1_idx, atom2_idx):
                internal_count += 1
                
        return internal_count
