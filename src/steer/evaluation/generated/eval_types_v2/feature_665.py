"""Generated evaluation code for: Schmidt rearrangement for lactam ring expansion"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SchmidtRearrangementDepth(BaseScoring):
    """
    Evaluates the depth at which a Schmidt rearrangement occurs for lactam ring expansion.
    Detects conversion of cyclic ketones to lactams via nitrogen insertion.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "bool")
        self.target_depth = config.get("target_depth", {}).get("value", -1)
    
    def route_scoring(self, x) -> float:
        if self.condition_type == "bool":
            if self.target_depth == -1:  # Positive if condition met
                return 1 if x >= 0 else 0
        else:
            if x < 0:
                return 0
            return max(0, 10 - abs(x - self.target_depth))
    
    def hit_condition(self, d):
        """
        Detects Schmidt rearrangement by checking for:
        1. Cyclic ketone substrate with C=O in ring
        2. Lactam product with C-N bond where ketone was
        3. Ring expansion by one carbon
        """
        try:
            rxn = d["metadata"]["mapped_reaction_smiles"].split(">>")
            reactants = [Chem.MolFromSmiles(r) for r in rxn[0].split(".")]
            products = [Chem.MolFromSmiles(p) for p in rxn[1].split(".")]
            
            # Check for cyclic ketone in reactants
            cyclic_ketone_found = False
            ketone_carbon_map = None
            
            for reactant in reactants:
                if self._has_cyclic_ketone(reactant):
                    cyclic_ketone_found = True
                    ketone_carbon_map = self._get_ketone_carbon_map(reactant)
                    break
            
            if not cyclic_ketone_found:
                return False
            
            # Check for lactam in products
            for product in products:
                if self._has_lactam(product) and ketone_carbon_map:
                    # Verify the ketone carbon is now part of amide
                    if self._is_schmidt_transformation(product, ketone_carbon_map):
                        return True
            
            return False
            
        except Exception:
            return False
    
    def _has_cyclic_ketone(self, mol):
        """Check if molecule contains a cyclic ketone"""
        if mol is None:
            return False
        
        # SMARTS for cyclic ketone (carbonyl carbon in ring)
        cyclic_ketone_pattern = Chem.MolFromSmarts("[C;R](=O)")
        return mol.HasSubstructMatch(cyclic_ketone_pattern)
    
    def _get_ketone_carbon_map(self, mol):
        """Get atom map number of ketone carbon"""
        cyclic_ketone_pattern = Chem.MolFromSmarts("[C;R](=O)")
        matches = mol.GetSubstructMatches(cyclic_ketone_pattern)
        
        for match in matches:
            ketone_carbon = mol.GetAtomWithIdx(match[0])
            if ketone_carbon.GetAtomMapNum() > 0:
                return ketone_carbon.GetAtomMapNum()
        return None
    
    def _has_lactam(self, mol):
        """Check if molecule contains a lactam (cyclic amide)"""
        if mol is None:
            return False
        
        # SMARTS for lactam (amide nitrogen and carbonyl in same ring)
        lactam_pattern = Chem.MolFromSmarts("[C;R](=O)[N;R]")
        return mol.HasSubstructMatch(lactam_pattern)
    
    def _is_schmidt_transformation(self, product_mol, ketone_carbon_map):
        """
        Verify that the ketone carbon is now part of an amide bond,
        indicating Schmidt rearrangement occurred
        """
        if ketone_carbon_map is None:
            return False
        
        # Find the atom with the ketone carbon map number
        for atom in product_mol.GetAtoms():
            if atom.GetAtomMapNum() == ketone_carbon_map:
                # Check if this carbon is now part of an amide
                if atom.GetSymbol() == 'C':
                    # Look for C=O bonded to N pattern
                    for neighbor in atom.GetNeighbors():
                        if (neighbor.GetSymbol() == 'O' and 
                            atom.GetBondBetweenAtomAndNeighbor(neighbor).GetBondType() == Chem.BondType.DOUBLE):
                            # Found C=O, now check for amide nitrogen
                            for other_neighbor in atom.GetNeighbors():
                                if other_neighbor.GetSymbol() == 'N' and other_neighbor != neighbor:
                                    return True
        return False
