"""Generated evaluation code for: Multi-step protecting group cycling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ProtectingGroupCycling(MultiRxnCondBase):
    """
    Evaluates multi-step protecting group cycling on carboxylic acid functionalities.
    Checks for sequences of protecting group transformations (protection/deprotection cycles)
    on the same substrate functionality.
    """
    
    def __init__(self, config):
        self.protection_sequences = config.get("protection_sequences", [])
        self.target_cycle_count = config.get("cycle_count", 3)
        self.substrate = config.get("substrate", "side_chain")
        
        # Define SMARTS patterns for different protecting groups
        self.pg_patterns = {
            "ethyl_ester": "[C:1](=[O:2])[O:3][CH2:4][CH3:5]",
            "benzyl_ester": "[C:1](=[O:2])[O:3][CH2:4]c1ccccc1",
            "tert_butyl_ester": "[C:1](=[O:2])[O:3][C:4]([CH3])([CH3])[CH3]",
            "free_acid": "[C:1](=[O:2])[OH:3]"
        }
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Track protecting group transformations
        pg_sequence = self.trace_protecting_group_sequence(reactions)
        cycle_count = self.count_protection_cycles(pg_sequence)
        
        # Check if we meet the target cycle count and sequence requirements
        condition_met = (
            cycle_count >= self.target_cycle_count and
            self.has_required_sequence(pg_sequence)
        )
        
        return condition_met, len(reactions)
    
    def trace_protecting_group_sequence(self, reactions):
        """Trace the sequence of protecting group transformations through the route."""
        pg_sequence = []
        
        for rxn in reactions:
            if self.is_protecting_group_reaction(rxn):
                # Determine which protecting groups are involved
                reactant_pgs = self.identify_protecting_groups(rxn["reactants"])
                product_pgs = self.identify_protecting_groups(rxn["products"])
                
                # Record the transformation
                transformation = {
                    "from": reactant_pgs,
                    "to": product_pgs,
                    "reaction": rxn
                }
                pg_sequence.append(transformation)
        
        return pg_sequence
    
    def is_protecting_group_reaction(self, rxn):
        """Check if a reaction involves protecting group chemistry."""
        # Look for ester formation/hydrolysis patterns
        ester_formation = "[C:1](=[O:2])[OH:3]>>[C:1](=[O:2])[O:3]"
        ester_hydrolysis = "[C:1](=[O:2])[O:3]>>[C:1](=[O:2])[OH:3]"
        
        rxn_smiles = rxn.get("reaction_smiles", "")
        
        # Check for ester transformations
        if "C(=O)O" in rxn_smiles and ("C(=O)OC" in rxn_smiles or "OC(C)" in rxn_smiles):
            return True
        
        return False
    
    def identify_protecting_groups(self, molecules):
        """Identify protecting groups present in a list of molecules."""
        found_groups = []
        
        for mol_smiles in molecules:
            try:
                mol = Chem.MolFromSmiles(mol_smiles)
                if mol is None:
                    continue
                    
                for pg_name, pattern in self.pg_patterns.items():
                    pg_mol = Chem.MolFromSmarts(pattern)
                    if pg_mol and mol.HasSubstructMatch(pg_mol):
                        found_groups.append(pg_name)
            except:
                continue
                
        return list(set(found_groups))  # Remove duplicates
    
    def count_protection_cycles(self, pg_sequence):
        """Count the number of protection/deprotection cycles."""
        if len(pg_sequence) < 2:
            return 0
        
        cycle_count = 0
        current_state = None
        
        for transformation in pg_sequence:
            # Identify if this is protection or deprotection
            from_groups = set(transformation["from"])
            to_groups = set(transformation["to"])
            
            # Protection: free acid -> protected ester
            if "free_acid" in from_groups and any(pg in to_groups for pg in ["ethyl_ester", "benzyl_ester", "tert_butyl_ester"]):
                if current_state == "deprotected":
                    cycle_count += 1
                current_state = "protected"
            
            # Deprotection: protected ester -> free acid or different ester
            elif any(pg in from_groups for pg in ["ethyl_ester", "benzyl_ester", "tert_butyl_ester"]):
                if "free_acid" in to_groups:
                    current_state = "deprotected"
                elif any(pg in to_groups for pg in ["ethyl_ester", "benzyl_ester", "tert_butyl_ester"]):
                    # Ester exchange counts as part of cycling strategy
                    if current_state == "protected":
                        cycle_count += 0.5  # Partial cycle for ester exchange
        
        return int(cycle_count)
    
    def has_required_sequence(self, pg_sequence):
        """Check if the sequence contains the required protecting group types."""
        if not self.protection_sequences:
            return True
        
        found_groups = set()
        for transformation in pg_sequence:
            found_groups.update(transformation["from"])
            found_groups.update(transformation["to"])
        
        # Check if all required protecting groups appear in the sequence
        required_groups = set(self.protection_sequences)
        return required_groups.issubset(found_groups)
