"""Generated evaluation code for: Early spiro-heterocycle formation via cycloaddition"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlySpiroHeterocycleFormation(BaseScoring):
    """
    Evaluates whether spiro-heterocycle formation via cycloaddition occurs early in the synthesis route.
    Checks for formation of spiro centers containing heteroatoms through cycloaddition reactions.
    """
    
    def __init__(self, config: Dict):
        self.timing_preference = config.get("timing", "early")  # "early" prefers formation at higher depth fractions
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Spiro-heterocycle formation doesn't occur
        else:
            if self.timing_preference == "early":
                return x  # Higher depth fraction (earlier in route) is better
            else:
                return 1 - x  # Lower depth fraction (later in route) is better
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction forms a spiro-heterocycle via cycloaddition.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants_smiles = rxn_parts[0]
            products_smiles = rxn_parts[1]
            
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            if not reactants or not products:
                return False
            
            # Check if this is a cycloaddition (typically 2 reactants -> 1 product with new ring)
            if not self._is_cycloaddition_pattern(reactants, products):
                return False
            
            # Check if spiro-heterocycle is formed
            return self._forms_spiro_heterocycle(reactants, products)
            
        except Exception:
            return False
    
    def _is_cycloaddition_pattern(self, reactants, products) -> bool:
        """
        Check if reaction pattern matches cycloaddition (multiple reactants combining to form cyclic product).
        """
        # Cycloaddition typically involves 2+ reactants forming a cyclic product
        if len(reactants) < 2:
            return False
        
        # Count total rings in reactants vs products
        reactant_rings = sum(mol.GetRingInfo().NumRings() for mol in reactants)
        product_rings = sum(mol.GetRingInfo().NumRings() for mol in products)
        
        # New ring should be formed
        return product_rings > reactant_rings
    
    def _forms_spiro_heterocycle(self, reactants, products) -> bool:
        """
        Check if the reaction forms a spiro center with heterocycles.
        """
        # Check each product for spiro-heterocycle formation
        for product in products:
            if self._has_spiro_heterocycle(product):
                # Verify this spiro center is newly formed (not present in reactants)
                if not any(self._has_spiro_heterocycle(reactant) for reactant in reactants):
                    return True
        return False
    
    def _has_spiro_heterocycle(self, mol) -> bool:
        """
        Detect spiro centers involving heterocycles.
        """
        if mol is None:
            return False
        
        ring_info = mol.GetRingInfo()
        if ring_info.NumRings() < 2:
            return False
        
        # Get all ring atoms
        ring_atoms_list = ring_info.AtomRings()
        
        # Check for spiro centers (atoms belonging to exactly 2 rings)
        for atom in mol.GetAtoms():
            atom_idx = atom.GetIdx()
            rings_containing_atom = [i for i, ring in enumerate(ring_atoms_list) 
                                   if atom_idx in ring]
            
            # Spiro center: atom in exactly 2 rings
            if len(rings_containing_atom) == 2:
                # Check if at least one of the rings contains heteroatoms
                ring1_atoms = ring_atoms_list[rings_containing_atom[0]]
                ring2_atoms = ring_atoms_list[rings_containing_atom[1]]
                
                ring1_has_hetero = any(mol.GetAtomWithIdx(idx).GetAtomicNum() not in [6, 1] 
                                     for idx in ring1_atoms)
                ring2_has_hetero = any(mol.GetAtomWithIdx(idx).GetAtomicNum() not in [6, 1] 
                                     for idx in ring2_atoms)
                
                if ring1_has_hetero or ring2_has_hetero:
                    return True
        
        return False
