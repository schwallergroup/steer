"""Generated evaluation code for: Linear functional group interconversion sequence"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LinearFunctionalGroupSequence(MultiRxnCondBase):
    """
    Evaluates routes for linear functional group interconversion sequences.
    Checks if the route contains a series of consecutive functional group 
    transformations on the same molecular scaffold in the specified order.
    """
    
    def __init__(self, config):
        self.strategy_type = config.get("strategy_type", "linear")
        self.sequential_transformations = config.get("sequential_transformations", [])
        
        # Define SMARTS patterns for functional groups
        self.fg_patterns = {
            "amine": "[NX3;H2,H1;!$(NC=O)]",
            "bromide": "[Br]",
            "nitrile": "[CX2]#[NX1]",
            "ketone": "[CX3]=[OX1]",
            "carboxylic_acid": "[CX3](=O)[OX1H0-,OX2H1]",
            "ester": "[CX3](=O)[OX2H0][#6]"
        }
    
    def condition_depth(self, d):
        """Check if the route contains the specified linear FG sequence"""
        reactions = self.get_rxns(d)
        
        if len(reactions) < len(self.sequential_transformations) - 1:
            return False, len(reactions)
        
        # Track functional group transformations in order
        sequence_found = self.detect_linear_sequence(reactions)
        
        return sequence_found, len(reactions)
    
    def detect_linear_sequence(self, reactions):
        """Detect if reactions contain the specified linear FG transformation sequence"""
        if len(self.sequential_transformations) < 2:
            return False
            
        # For each possible starting position in the reaction sequence
        for start_idx in range(len(reactions) - len(self.sequential_transformations) + 2):
            if self.check_sequence_at_position(reactions, start_idx):
                return True
        return False
    
    def check_sequence_at_position(self, reactions, start_idx):
        """Check if the FG sequence starts at the given reaction position"""
        sequence_matches = 0
        required_matches = len(self.sequential_transformations) - 1
        
        for i in range(required_matches):
            if start_idx + i >= len(reactions):
                break
                
            rxn = reactions[start_idx + i]
            from_fg = self.sequential_transformations[i]
            to_fg = self.sequential_transformations[i + 1]
            
            if self.detect_fg_transformation(rxn, from_fg, to_fg):
                sequence_matches += 1
            else:
                break
        
        return sequence_matches == required_matches
    
    def detect_fg_transformation(self, rxn, from_fg, to_fg):
        """Check if a reaction transforms from_fg to to_fg"""
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            products = rxn_parts[0]
            reactants = rxn_parts[1]
            
            # Parse molecules
            prod_mol = Chem.MolFromSmiles(products)
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            
            if not prod_mol or not all(reactant_mols):
                return False
            
            # Check if product has the target functional group
            to_pattern = Chem.MolFromSmarts(self.fg_patterns[to_fg])
            if not prod_mol.HasSubstructMatch(to_pattern):
                return False
            
            # Check if any reactant has the source functional group
            from_pattern = Chem.MolFromSmarts(self.fg_patterns[from_fg])
            reactant_has_from_fg = any(mol.HasSubstructMatch(from_pattern) for mol in reactant_mols)
            
            if not reactant_has_from_fg:
                return False
            
            # Ensure the transformation is on the same scaffold by checking
            # that the core structure is preserved (simplified check)
            return self.same_core_scaffold(prod_mol, reactant_mols[0])
            
        except Exception:
            return False
    
    def same_core_scaffold(self, prod_mol, main_reactant):
        """Simple check if molecules share the same core scaffold"""
        try:
            # Remove functional groups and compare core structures
            prod_core = self.get_core_structure(prod_mol)
            reactant_core = self.get_core_structure(main_reactant)
            
            return prod_core == reactant_core
        except Exception:
            return True  # Default to True if comparison fails
    
    def get_core_structure(self, mol):
        """Get core structure by removing common functional groups"""
        try:
            # Create a copy and remove common FG atoms for comparison
            mol_copy = Chem.Mol(mol)
            
            # Simple approach: count heavy atoms and rings as core signature
            num_heavy_atoms = mol_copy.GetNumHeavyAtoms()
            ring_info = mol_copy.GetRingInfo()
            num_rings = ring_info.NumRings()
            
            return (num_heavy_atoms, num_rings)
        except Exception:
            return (0, 0)
