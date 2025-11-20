"""Generated evaluation code for: Multiple protecting group swaps on phenolic positions"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class MultipleProtectingGroupSwaps(MultiRxnCondBase):
    """
    Evaluates synthesis routes for multiple protecting group swaps on phenolic positions.
    Detects sequences like MOM→Me and Ac→Bn→Me transformations and penalizes routes
    with unnecessary protecting group changes.
    """
    
    def __init__(self, config):
        self.swap_count = config.get("swap_count", 2)
        self.functional_group = config.get("functional_group", "phenol")
        self.swap_sequences = config.get("swap_sequences", [])
        
        # Define SMARTS patterns for protecting groups
        self.protecting_patterns = {
            "phenol": "[OH1][c]",  # Free phenol
            "MOM": "[O][CH2][O][CH3]",  # Methoxymethyl ether
            "methyl": "[O][CH3]",  # Methyl ether (when connected to aromatic)
            "acetate": "[O]C(=O)[CH3]",  # Acetate ester
            "benzyl": "[O][CH2]c1ccccc1",  # Benzyl ether
        }
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        detected_swaps = self.detect_protecting_group_swaps(reactions)
        
        # Check if we have the required number of swap sequences
        condition_met = len(detected_swaps) >= self.swap_count
        
        return condition_met, len(reactions)
    
    def detect_protecting_group_swaps(self, reactions):
        """
        Analyze reactions to detect protecting group swap sequences.
        Returns list of detected swap sequences.
        """
        detected_swaps = []
        
        # Track protecting group changes across reactions
        pg_changes = []
        for rxn in reactions:
            changes = self.analyze_protecting_group_changes(rxn)
            if changes:
                pg_changes.extend(changes)
        
        # Look for swap sequences
        for sequence_name in self.swap_sequences:
            if self.detect_swap_sequence(pg_changes, sequence_name):
                detected_swaps.append(sequence_name)
        
        return detected_swaps
    
    def analyze_protecting_group_changes(self, rxn):
        """
        Analyze a single reaction for protecting group transformations.
        Returns list of (from_pg, to_pg, position) tuples.
        """
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return []
        
        reactants = [Chem.MolFromSmiles(s.strip()) for s in rxn_parts[0].split(".") if s.strip()]
        products = [Chem.MolFromSmiles(s.strip()) for s in rxn_parts[1].split(".") if s.strip()]
        
        if not all(reactants) or not all(products):
            return []
        
        changes = []
        
        # Find molecules with phenolic positions and track PG changes
        for react_mol in reactants:
            for prod_mol in products:
                if self.molecules_similar(react_mol, prod_mol):
                    pg_changes = self.compare_protecting_groups(react_mol, prod_mol)
                    changes.extend(pg_changes)
        
        return changes
    
    def molecules_similar(self, mol1, mol2):
        """
        Check if two molecules are similar enough to be the same scaffold
        with different protecting groups.
        """
        if mol1 is None or mol2 is None:
            return False
        
        # Simple check: similar molecular weight and atom count
        mw_diff = abs(Chem.rdMolDescriptors.CalcExactMolWt(mol1) - 
                     Chem.rdMolDescriptors.CalcExactMolWt(mol2))
        atom_diff = abs(mol1.GetNumHeavyAtoms() - mol2.GetNumHeavyAtoms())
        
        # Allow some difference for protecting group changes
        return mw_diff < 200 and atom_diff < 10
    
    def compare_protecting_groups(self, mol1, mol2):
        """
        Compare protecting groups between two similar molecules.
        """
        changes = []
        
        # Get phenolic positions and their protecting groups
        phenols1 = self.get_phenolic_protecting_groups(mol1)
        phenols2 = self.get_phenolic_protecting_groups(mol2)
        
        # Match positions and detect changes
        for pos1, pg1 in phenols1.items():
            for pos2, pg2 in phenols2.items():
                if self.positions_match(mol1, pos1, mol2, pos2) and pg1 != pg2:
                    changes.append((pg1, pg2, pos1))
        
        return changes
    
    def get_phenolic_protecting_groups(self, mol):
        """
        Identify phenolic positions and their protecting groups.
        Returns dict of {position: protecting_group_type}.
        """
        if mol is None:
            return {}
        
        phenolic_positions = {}
        
        # Find aromatic carbons connected to oxygen
        aromatic_c_o_pattern = Chem.MolFromSmarts("[c][O]")
        matches = mol.GetSubstructMatches(aromatic_c_o_pattern)
        
        for match in matches:
            c_idx, o_idx = match
            pg_type = self.identify_protecting_group(mol, o_idx)
            phenolic_positions[c_idx] = pg_type
        
        return phenolic_positions
    
    def identify_protecting_group(self, mol, oxygen_idx):
        """
        Identify the type of protecting group attached to an oxygen atom.
        """
        atom = mol.GetAtomWithIdx(oxygen_idx)
        
        # Check neighbors of oxygen
        neighbors = [mol.GetAtomWithIdx(n.GetIdx()) for n in atom.GetNeighbors()]
        
        # Check for different protecting group patterns
        for pg_name, pattern in self.protecting_patterns.items():
            if pg_name == "phenol":
                continue
            
            pg_mol = Chem.MolFromSmarts(pattern)
            if pg_mol and mol.HasSubstructMatch(pg_mol):
                matches = mol.GetSubstructMatches(pg_mol)
                for match in matches:
                    if oxygen_idx in match:
                        return pg_name
        
        # Default to free phenol if no protecting group found
        return "phenol"
    
    def positions_match(self, mol1, pos1, mol2, pos2):
        """
        Simple heuristic to check if positions in different molecules correspond.
        """
        # This is a simplified approach - in practice, you might need more sophisticated
        # atom mapping or structural alignment
        return True
    
    def detect_swap_sequence(self, pg_changes, sequence_name):
        """
        Detect if a specific swap sequence occurs in the protecting group changes.
        """
        if sequence_name == "MOM_to_methyl":
            return any(change[0] == "MOM" and change[1] == "methyl" for change in pg_changes)
        
        elif sequence_name == "acetate_to_benzyl_to_methyl":
            # Look for acetate→benzyl and benzyl→methyl changes
            has_ac_to_bn = any(change[0] == "acetate" and change[1] == "benzyl" for change in pg_changes)
            has_bn_to_me = any(change[0] == "benzyl" and change[1] == "methyl" for change in pg_changes)
            return has_ac_to_bn and has_bn_to_me
        
        return False
