"""Generated evaluation code for: Early bicyclic core formation via cycloaddition"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlyBicyclicCycloaddition(BaseScoring):
    """
    Evaluates synthesis routes for early bicyclic core formation via cycloaddition.
    Detects cycloaddition reactions that form exactly 2 rings simultaneously,
    with better scores for earlier occurrence in the synthesis.
    """
    
    def __init__(self, config: Dict):
        self.target_rings = config["parameters"]["rings_formed"]
        self.reaction_type = config["parameters"]["reaction_type"]
        self.timing = config["parameters"]["timing"]
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Condition not met
        
        if self.timing == "early":
            return 1 - x  # Earlier is better (depth fraction closer to 0)
        elif self.timing == "late":
            return x  # Later is better (depth fraction closer to 1)
        else:
            return 1  # Any occurrence is good
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction node represents a bicyclic cycloaddition"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            prod_smiles, react_smiles = mapped_rxn.split(">>")
            product = Chem.MolFromSmiles(prod_smiles)
            reactants = [Chem.MolFromSmiles(r) for r in react_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check if this is a cycloaddition by verifying:
            # 1. Number of rings increases by exactly target_rings
            # 2. Reaction involves formation of new C-C or C-heteroatom bonds in cyclic fashion
            
            reactant_rings = sum(self._count_rings(mol) for mol in reactants)
            product_rings = self._count_rings(product)
            
            rings_formed = product_rings - reactant_rings
            
            if rings_formed != self.target_rings:
                return False
            
            # Verify it's a cycloaddition by checking for characteristic patterns
            return self._is_cycloaddition_pattern(reactants, product)
            
        except Exception:
            return False
    
    def _count_rings(self, mol) -> int:
        """Count the number of rings in a molecule"""
        if mol is None:
            return 0
        return mol.GetRingInfo().NumRings()
    
    def _is_cycloaddition_pattern(self, reactants, product) -> bool:
        """
        Check if the reaction follows cycloaddition patterns.
        Look for formation of new bonds between atoms that were in different reactants.
        """
        # For a true cycloaddition, we expect:
        # - Two or more reactants combining
        # - Formation of new sigma bonds creating rings
        # - Characteristic atom mapping pattern
        
        if len(reactants) < 2:
            return False
        
        # Check for common cycloaddition substructure patterns in product
        # These patterns indicate bicyclic systems commonly formed by cycloadditions
        bicyclic_patterns = [
            # Fused 6-6 rings (Diels-Alder products)
            "C1=CC=CC=C1C2CCCCC2",
            "C1CCC2CCCCC2C1",
            # Fused 5-6 rings
            "C1CCC2CCCC2C1",
            "C1=CC=C2CCCC2C1",
            # Bridged bicyclic systems
            "C1CC2CCC1CC2",
            "C1CCC2CC1CC2",
            # Bicyclic with heteroatoms (common in 1,3-dipolar cycloadditions)
            "C1CC2NCCN2C1",
            "C1CC2OCCN2C1"
        ]
        
        for pattern in bicyclic_patterns:
            pattern_mol = Chem.MolFromSmarts(pattern)
            if pattern_mol and product.HasSubstructMatch(pattern_mol):
                return True
        
        # Additional check: verify the reaction creates exactly 2 new rings
        # by examining bond formation patterns
        return self._check_bond_formation_pattern(reactants, product)
    
    def _check_bond_formation_pattern(self, reactants, product) -> bool:
        """
        Verify that new bonds are formed in a pattern consistent with cycloaddition
        forming exactly 2 rings.
        """
        try:
            # Get atom mappings to track which atoms form new bonds
            product_atom_maps = {atom.GetAtomMapNum(): atom.GetIdx() 
                               for atom in product.GetAtoms() if atom.GetAtomMapNum() > 0}
            
            reactant_atom_maps = {}
            for mol_idx, mol in enumerate(reactants):
                for atom in mol.GetAtoms():
                    if atom.GetAtomMapNum() > 0:
                        reactant_atom_maps[atom.GetAtomMapNum()] = (mol_idx, atom.GetIdx())
            
            # Count new bonds formed between atoms from different reactants
            new_bonds = 0
            for bond in product.GetBonds():
                atom1_map = product.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetAtomMapNum()
                atom2_map = product.GetAtomWithIdx(bond.GetEndAtomIdx()).GetAtomMapNum()
                
                if atom1_map > 0 and atom2_map > 0:
                    if (atom1_map in reactant_atom_maps and atom2_map in reactant_atom_maps):
                        mol1_idx = reactant_atom_maps[atom1_map][0]
                        mol2_idx = reactant_atom_maps[atom2_map][0]
                        
                        # Check if this bond existed in reactants
                        if mol1_idx != mol2_idx:  # Atoms from different reactants
                            new_bonds += 1
                        elif mol1_idx == mol2_idx:  # Same reactant - check if bond existed
                            reactant_mol = reactants[mol1_idx]
                            atom1_idx = reactant_atom_maps[atom1_map][1]
                            atom2_idx = reactant_atom_maps[atom2_map][1]
                            if not reactant_mol.GetBondBetweenAtoms(atom1_idx, atom2_idx):
                                new_bonds += 1
            
            # Cycloadditions forming 2 rings typically form 3-4 new bonds
            return 3 <= new_bonds <= 4
            
        except Exception:
            return False
