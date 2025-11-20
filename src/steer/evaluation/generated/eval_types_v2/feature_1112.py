"""Generated evaluation code for: Circular protecting group strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class CircularProtectingGroupStrategy(MultiRxnCondBase):
    """
    Detects circular protecting group strategies where functional groups are 
    transformed in a cycle (e.g., OH->Cl->NH2->OH) at the same molecular position.
    """
    
    def __init__(self, config):
        self.strategy_type = config.get("strategy_type", "circular")
        self.functional_groups = config.get("functional_groups", ["OH", "Cl", "NH2"])
        self.cycle_detected = config.get("cycle_detected", True)
        
        # Define SMARTS patterns for functional groups
        self.fg_patterns = {
            "OH": "[OH1]",
            "Cl": "[Cl]", 
            "NH2": "[NH2]",
            "Br": "[Br]",
            "I": "[I]",
            "OAc": "[O][C](=O)[CH3]",
            "OTs": "[O]S(=O)(=O)[c]",
            "COOH": "[C](=O)[OH]",
            "COOEt": "[C](=O)[O][CH2][CH3]"
        }
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Track transformations by atom map number
        transformations = self.track_transformations(reactions)
        
        # Check for circular patterns
        has_cycle = self.detect_circular_pattern(transformations)
        
        condition_met = has_cycle == self.cycle_detected
        return condition_met, len(reactions)
    
    def track_transformations(self, reactions):
        """Track functional group transformations by atom map number across reactions."""
        transformations = {}  # {atom_map: [fg1, fg2, fg3, ...]}
        
        for rxn_smiles in reactions:
            try:
                reactants_smi, products_smi = rxn_smiles.split(">>")
                reactants = [Chem.MolFromSmiles(smi) for smi in reactants_smi.split(".")]
                products = [Chem.MolFromSmiles(smi) for smi in products_smi.split(".")]
                
                # Get functional groups for each mapped atom in reactants and products
                reactant_fgs = self.get_mapped_functional_groups(reactants)
                product_fgs = self.get_mapped_functional_groups(products)
                
                # Track changes
                for atom_map in set(reactant_fgs.keys()) | set(product_fgs.keys()):
                    if atom_map not in transformations:
                        transformations[atom_map] = []
                    
                    # Add transformation if functional group changed
                    react_fg = reactant_fgs.get(atom_map)
                    prod_fg = product_fgs.get(atom_map)
                    
                    if react_fg and prod_fg and react_fg != prod_fg:
                        transformations[atom_map].extend([react_fg, prod_fg])
                    elif react_fg and not prod_fg:
                        transformations[atom_map].append(react_fg)
                    elif not react_fg and prod_fg:
                        transformations[atom_map].append(prod_fg)
                        
            except Exception:
                continue
                
        return transformations
    
    def get_mapped_functional_groups(self, molecules):
        """Get functional groups for each mapped atom in molecules."""
        mapped_fgs = {}
        
        for mol in molecules:
            if mol is None:
                continue
                
            for atom in mol.GetAtoms():
                map_num = atom.GetAtomMapNum()
                if map_num == 0:
                    continue
                    
                # Check what functional group this atom is part of
                fg = self.identify_functional_group(mol, atom)
                if fg:
                    mapped_fgs[map_num] = fg
                    
        return mapped_fgs
    
    def identify_functional_group(self, mol, atom):
        """Identify functional group containing the given atom."""
        atom_idx = atom.GetIdx()
        
        for fg_name, pattern in self.fg_patterns.items():
            try:
                patt_mol = Chem.MolFromSmarts(pattern)
                if patt_mol and mol.HasSubstructMatch(patt_mol):
                    matches = mol.GetSubstructMatches(patt_mol)
                    for match in matches:
                        if atom_idx in match:
                            return fg_name
            except Exception:
                continue
                
        return None
    
    def detect_circular_pattern(self, transformations):
        """Detect if any atom undergoes circular functional group transformations."""
        target_fgs = set(self.functional_groups)
        
        for atom_map, fg_sequence in transformations.items():
            if len(fg_sequence) < 3:  # Need at least 3 groups for a cycle
                continue
                
            # Check if sequence contains target functional groups
            seq_fgs = set(fg_sequence)
            if not seq_fgs.issubset(target_fgs):
                continue
                
            # Look for cycles: same FG appears at start and end with others in between
            for i in range(len(fg_sequence) - 2):
                for j in range(i + 2, len(fg_sequence)):
                    if fg_sequence[i] == fg_sequence[j]:
                        cycle_fgs = set(fg_sequence[i:j+1])
                        # Check if cycle contains the target functional groups
                        if len(cycle_fgs.intersection(target_fgs)) >= 2:
                            return True
                            
        return False
