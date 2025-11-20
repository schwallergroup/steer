"""Generated evaluation code for: Late stage quaternary center formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageQuaternaryCenterFormation(BaseScoring):
    """
    Evaluates routes for late-stage quaternary center formation via enolate alkylation.
    Rewards routes where a quaternary carbon is formed late in the synthesis.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "continuous")
        self.target_depth = config.get("target_depth", {}).get("value", 0.8)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Quaternary center formation doesn't happen
        else:
            return 1 - x  # Later formation is better (closer to 1.0 depth)
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction forms a quaternary carbon via enolate alkylation"""
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            reactants_smiles, product_smiles = rxn_smiles.split(">>")
            
            # Parse molecules
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check if product has quaternary carbon
            quaternary_carbons = self._find_quaternary_carbons(product)
            if not quaternary_carbons:
                return False
            
            # Check if any reactant lacks these quaternary carbons (i.e., they're formed in this step)
            for quat_atom_map in quaternary_carbons:
                if self._is_newly_formed_quaternary(quat_atom_map, reactants, product):
                    # Check if it's via enolate alkylation pattern
                    if self._is_enolate_alkylation(quat_atom_map, reactants, product):
                        return True
            
            return False
            
        except Exception:
            return False
    
    def _find_quaternary_carbons(self, mol):
        """Find quaternary carbons (carbon with 4 non-hydrogen substituents)"""
        quaternary_atoms = []
        for atom in mol.GetAtoms():
            if (atom.GetAtomicNum() == 6 and  # Carbon
                atom.GetDegree() == 4 and     # 4 bonds
                atom.GetTotalNumHs() == 0):   # No hydrogens
                quaternary_atoms.append(atom.GetAtomMapNum())
        return quaternary_atoms
    
    def _is_newly_formed_quaternary(self, quat_atom_map, reactants, product):
        """Check if quaternary carbon is newly formed in this reaction"""
        # Find the atom in reactants with same map number
        for reactant in reactants:
            for atom in reactant.GetAtoms():
                if atom.GetAtomMapNum() == quat_atom_map:
                    # If this carbon exists in reactant but is not quaternary, it's newly formed
                    if (atom.GetAtomicNum() == 6 and 
                        not (atom.GetDegree() == 4 and atom.GetTotalNumHs() == 0)):
                        return True
        return False
    
    def _is_enolate_alkylation(self, quat_atom_map, reactants, product):
        """Check if quaternary center formation follows enolate alkylation pattern"""
        # Look for carbonyl adjacent to quaternary center (enolate precursor)
        product_atom = None
        for atom in product.GetAtoms():
            if atom.GetAtomMapNum() == quat_atom_map:
                product_atom = atom
                break
        
        if not product_atom:
            return False
        
        # Check if quaternary carbon is adjacent to carbonyl
        for neighbor in product_atom.GetNeighbors():
            if neighbor.GetAtomicNum() == 6:  # Carbon neighbor
                for neighbor2 in neighbor.GetNeighbors():
                    if (neighbor2.GetAtomicNum() == 8 and  # Oxygen
                        product.GetBondBetweenAtoms(neighbor.GetIdx(), neighbor2.GetIdx()).GetBondType() == Chem.BondType.DOUBLE):
                        return True
        
        # Also check for direct attachment to carbonyl carbon
        for neighbor in product_atom.GetNeighbors():
            if neighbor.GetAtomicNum() == 6:  # Carbon
                for neighbor2 in neighbor.GetNeighbors():
                    if (neighbor2.GetAtomicNum() == 8 and  # Oxygen 
                        product.GetBondBetweenAtoms(neighbor.GetIdx(), neighbor2.GetIdx()).GetBondType() == Chem.BondType.DOUBLE):
                        return True
        
        return False
