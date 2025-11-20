"""Generated evaluation code for: Sequential protecting group strategy with Boc and benzyl"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SequentialProtectingGroupStrategy(MultiRxnCondBase):
    """
    Evaluates routes for sequential use of Boc and benzyl protecting groups.
    Checks that both protection types are used and that they follow a coordinated
    sequence for selective functionalization.
    """
    
    def __init__(self, config):
        self.pg_types = config["parameters"]["pg_types"]
        self.strategy = config["parameters"]["strategy"]
        
        # Define SMARTS patterns for detecting protecting groups
        self.boc_patterns = [
            "[NH1,NH2,NH0]-C(=O)-O-C(C)(C)C",  # Boc protection
            "C(=O)-O-C(C)(C)C",  # Boc group general
        ]
        
        self.benzyl_patterns = [
            "c1ccccc1-[CH2]-O-[#6]",  # Benzyl ether
            "c1ccccc1-[CH2]-[NH,NH2]",  # Benzyl amine
            "c1ccccc1-[CH2]-[OH]",  # Benzyl alcohol intermediate
        ]
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        boc_reactions = []
        benzyl_reactions = []
        
        for i, rxn in enumerate(reactions):
            if self.detect_boc_protection(rxn):
                boc_reactions.append(i)
            if self.detect_benzyl_protection(rxn):
                benzyl_reactions.append(i)
        
        # Check if both protecting groups are present
        has_boc = len(boc_reactions) > 0
        has_benzyl = len(benzyl_reactions) > 0
        both_present = has_boc and has_benzyl
        
        if not both_present:
            return False, len(reactions)
        
        # Check sequential strategy - reactions should not overlap significantly
        # and should show coordinated use
        sequential_condition = self.check_sequential_strategy(
            boc_reactions, benzyl_reactions, len(reactions)
        )
        
        return sequential_condition, len(reactions)
    
    def detect_boc_protection(self, rxn):
        """Detect Boc protection/deprotection reactions"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        # Check for Boc installation (reactant without Boc -> product with Boc)
        reactant_mols = [Chem.MolFromSmiles(r) for r in reactants.split(".")]
        product_mols = [Chem.MolFromSmiles(p) for p in products.split(".")]
        
        if not all(reactant_mols) or not all(product_mols):
            return False
        
        reactant_has_boc = any(self.has_boc_pattern(mol) for mol in reactant_mols if mol)
        product_has_boc = any(self.has_boc_pattern(mol) for mol in product_mols if mol)
        
        # Boc installation or removal indicates Boc strategy
        return reactant_has_boc != product_has_boc
    
    def detect_benzyl_protection(self, rxn):
        """Detect benzyl protection/deprotection reactions"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        reactant_mols = [Chem.MolFromSmiles(r) for r in reactants.split(".")]
        product_mols = [Chem.MolFromSmiles(p) for p in products.split(".")]
        
        if not all(reactant_mols) or not all(product_mols):
            return False
        
        reactant_has_benzyl = any(self.has_benzyl_pattern(mol) for mol in reactant_mols if mol)
        product_has_benzyl = any(self.has_benzyl_pattern(mol) for mol in product_mols if mol)
        
        # Benzyl installation or removal indicates benzyl strategy
        return reactant_has_benzyl != product_has_benzyl
    
    def has_boc_pattern(self, mol):
        """Check if molecule contains Boc protecting group"""
        if not mol:
            return False
        for pattern in self.boc_patterns:
            patt_mol = Chem.MolFromSmarts(pattern)
            if patt_mol and mol.HasSubstructMatch(patt_mol):
                return True
        return False
    
    def has_benzyl_pattern(self, mol):
        """Check if molecule contains benzyl protecting group"""
        if not mol:
            return False
        for pattern in self.benzyl_patterns:
            patt_mol = Chem.MolFromSmarts(pattern)
            if patt_mol and mol.HasSubstructMatch(patt_mol):
                return True
        return False
    
    def check_sequential_strategy(self, boc_reactions, benzyl_reactions, total_reactions):
        """
        Check if the protecting groups are used in a sequential/coordinated manner
        rather than randomly throughout the synthesis
        """
        if not boc_reactions or not benzyl_reactions:
            return False
        
        # For sequential strategy, we expect some separation between the two types
        # or coordinated use where one type is clustered in certain regions
        all_pg_reactions = sorted(boc_reactions + benzyl_reactions)
        
        # Check if there's some strategic separation or clustering
        # Simple heuristic: if reactions are not completely interleaved
        boc_set = set(boc_reactions)
        benzyl_set = set(benzyl_reactions)
        
        # Count transitions between protection types
        transitions = 0
        last_type = None
        
        for rxn_idx in all_pg_reactions:
            current_type = "boc" if rxn_idx in boc_set else "benzyl"
            if last_type and last_type != current_type:
                transitions += 1
            last_type = current_type
        
        # Sequential strategy should have fewer transitions relative to total PG reactions
        max_expected_transitions = len(all_pg_reactions) // 2
        
        return transitions <= max_expected_transitions
