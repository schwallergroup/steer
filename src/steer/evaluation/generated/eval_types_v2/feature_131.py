"""Generated evaluation code for: Cyclic functional group interconversion via multiple intermediates"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class CyclicFunctionalGroupInterconversion(MultiRxnCondBase):
    """
    Evaluates synthesis routes that perform cyclic functional group interconversion
    by converting a starting functional group to an ending group via multiple 
    intermediate steps at the same position.
    """
    
    def __init__(self, config):
        self.start_group = config["start_group"]
        self.end_group = config["end_group"]
        self.intermediate_steps = config["intermediate_steps"]
        self.same_position = config.get("same_position", True)
        
        # Define SMARTS patterns for functional groups
        self.group_patterns = {
            "NO2": "[N+](=O)[O-]",
            "NH2": "[NH2]",
            "Br": "[Br]",
            "COOMe": "C(=O)O[CH3]",
            "COOH": "C(=O)[OH]",
            "CN": "C#N",
            "CHO": "C=O",
            "OH": "[OH]"
        }
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Track functional group transformations through the route
        transformations = []
        position_tracker = {}
        
        for rxn in reactions:
            transformation = self.detect_functional_group_change(rxn)
            if transformation:
                transformations.append(transformation)
        
        # Check if we have the required cyclic interconversion
        condition = self.has_cyclic_interconversion(transformations)
        return condition, len(reactions)
    
    def detect_functional_group_change(self, rxn):
        """Detect functional group changes in a reaction"""
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return None
                
            product = Chem.MolFromSmiles(rxn_parts[0])
            reactants = [Chem.MolFromSmiles(r) for r in rxn_parts[1].split(".")]
            
            if not product or not all(reactants):
                return None
            
            # Find functional groups in product and reactants
            prod_groups = self.find_functional_groups(product)
            react_groups = []
            for reactant in reactants:
                react_groups.extend(self.find_functional_groups(reactant))
            
            # Detect changes using atom mapping if same_position is True
            if self.same_position:
                return self.detect_position_specific_change(product, reactants, rxn)
            else:
                # Just look for appearance/disappearance of functional groups
                for group in self.group_patterns.keys():
                    if group in react_groups and group not in prod_groups:
                        return {"from": group, "to": None}
                    elif group not in react_groups and group in prod_groups:
                        return {"from": None, "to": group}
            
            return None
            
        except Exception:
            return None
    
    def find_functional_groups(self, mol):
        """Find functional groups present in a molecule"""
        groups_found = []
        for group_name, pattern in self.group_patterns.items():
            try:
                pattern_mol = Chem.MolFromSmarts(pattern)
                if pattern_mol and mol.HasSubstructMatch(pattern_mol):
                    groups_found.append(group_name)
            except Exception:
                continue
        return groups_found
    
    def detect_position_specific_change(self, product, reactants, rxn):
        """Detect functional group changes at the same atom position using mapping"""
        try:
            # Get atom mapping for product
            prod_map = {atom.GetAtomMapNum(): idx for idx, atom in enumerate(product.GetAtoms()) 
                       if atom.GetAtomMapNum() > 0}
            
            # Find mapped positions that have functional group changes
            for map_num, prod_idx in prod_map.items():
                prod_atom = product.GetAtom(prod_idx)
                prod_env = self.get_atom_environment(product, prod_idx)
                
                # Check corresponding atoms in reactants
                for reactant in reactants:
                    react_map = {atom.GetAtomMapNum(): idx for idx, atom in enumerate(reactant.GetAtoms())}
                    if map_num in react_map:
                        react_idx = react_map[map_num]
                        react_env = self.get_atom_environment(reactant, react_idx)
                        
                        # Compare environments to detect functional group change
                        prod_group = self.classify_environment(prod_env)
                        react_group = self.classify_environment(react_env)
                        
                        if prod_group != react_group and prod_group and react_group:
                            return {"from": react_group, "to": prod_group, "position": map_num}
            
            return None
            
        except Exception:
            return None
    
    def get_atom_environment(self, mol, atom_idx):
        """Get the local environment around an atom"""
        atom = mol.GetAtom(atom_idx)
        neighbors = []
        for neighbor in atom.GetNeighbors():
            bond = mol.GetBondBetweenAtoms(atom_idx, neighbor.GetIdx())
            neighbors.append((neighbor.GetSymbol(), bond.GetBondType()))
        return sorted(neighbors)
    
    def classify_environment(self, environment):
        """Classify atom environment to functional group"""
        env_str = str(environment)
        
        # Simple heuristics for functional group classification
        if "('O', BondType.DOUBLE)" in env_str and "('O', BondType.SINGLE)" in env_str:
            if "('C', BondType.SINGLE)" in env_str:
                return "COOH" if "OH" in env_str else "COOMe"
        elif "('N', BondType.SINGLE)" in env_str and environment.count(('H', None)) >= 1:
            return "NH2"
        elif "('O', BondType.DOUBLE)" in env_str and environment.count(('O', None)) == 2:
            return "NO2"
        elif "('Br', BondType.SINGLE)" in env_str:
            return "Br"
            
        return None
    
    def has_cyclic_interconversion(self, transformations):
        """Check if transformations represent cyclic interconversion"""
        if not transformations:
            return False
        
        # Extract sequence of functional groups
        sequence = []
        for trans in transformations:
            if trans.get("from"):
                sequence.append(trans["from"])
            if trans.get("to"):
                sequence.append(trans["to"])
        
        if len(sequence) < 3:  # Need at least start -> intermediate -> end
            return False
        
        # Check if we start and end with the expected groups
        has_start = self.start_group in sequence
        has_end = self.end_group in sequence
        
        # Check if we visit required intermediates
        has_intermediates = all(step in sequence for step in self.intermediate_steps)
        
        # For true cyclic interconversion, we should return to a group we've seen
        has_cycle = len(set(sequence)) < len(sequence)
        
        return has_start and has_end and has_intermediates and has_cycle
