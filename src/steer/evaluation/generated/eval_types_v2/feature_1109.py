"""Generated evaluation code for: Late spiro-cyclopropane ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SpiroCyclopropaneFormation(BaseScoring):
    """
    Evaluates late-stage spiro-cyclopropane ring formation in synthesis routes.
    Detects when a cyclopropane ring is formed that creates a spiro center.
    """
    
    def __init__(self, config: Dict):
        self.cyclopropane_pattern = Chem.MolFromSmarts("C1CC1")
        self.timing = config["parameters"].get("timing", "late")
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Formation doesn't happen
        else:
            # Later formation is better for late timing preference
            return 1 - x
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction forms a spiro-cyclopropane ring"""
        rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
        rxn_parts = rxn_smiles.split(">>")
        
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        # Parse reactants and products
        reactant_mols = []
        for r_smi in reactants.split("."):
            mol = Chem.MolFromSmiles(r_smi)
            if mol:
                reactant_mols.append(mol)
        
        product_mols = []
        for p_smi in products.split("."):
            mol = Chem.MolFromSmiles(p_smi)
            if mol:
                product_mols.append(mol)
        
        # Count cyclopropane rings in reactants vs products
        reactant_cycloprop_count = sum(
            len(mol.GetSubstructMatches(self.cyclopropane_pattern)) 
            for mol in reactant_mols
        )
        
        product_cycloprop_count = sum(
            len(mol.GetSubstructMatches(self.cyclopropane_pattern)) 
            for mol in product_mols
        )
        
        # Check if cyclopropane ring was formed
        if product_cycloprop_count <= reactant_cycloprop_count:
            return False
        
        # Check if the formed cyclopropane is part of a spiro system
        for mol in product_mols:
            if self._has_spiro_cyclopropane(mol):
                return True
                
        return False
    
    def _has_spiro_cyclopropane(self, mol) -> bool:
        """Check if molecule contains a spiro-cyclopropane system"""
        cycloprop_matches = mol.GetSubstructMatches(self.cyclopropane_pattern)
        
        for match in cycloprop_matches:
            # Check each atom in the cyclopropane ring
            for atom_idx in match:
                atom = mol.GetAtomWithIdx(atom_idx)
                
                # Count rings this atom participates in
                ring_info = mol.GetRingInfo()
                atom_rings = [ring for ring in ring_info.AtomRings() if atom_idx in ring]
                
                # Spiro center: atom in at least 2 rings, not sharing edges
                if len(atom_rings) >= 2:
                    # Check if rings share only this atom (spiro condition)
                    for i, ring1 in enumerate(atom_rings):
                        for ring2 in atom_rings[i+1:]:
                            shared_atoms = set(ring1) & set(ring2)
                            if len(shared_atoms) == 1 and atom_idx in shared_atoms:
                                return True
                                
        return False
