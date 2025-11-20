"""Generated evaluation code for: Sequential protecting group cycling strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ProtectingGroupCycling(MultiRxnCondBase):
    """
    Evaluates sequential protecting group cycling strategy.
    Checks if specified protecting groups (Boc, DPM) are installed and removed
    in sequence throughout the synthesis route.
    """
    
    def __init__(self, config):
        self.protection_groups = config["parameters"]["protection_deprotection_pairs"]
        self.sequential = config["parameters"]["sequential"]
        
        # SMARTS patterns for protecting groups
        self.pg_patterns = {
            "Boc": "[NX3][CX3](=[OX1])[OX2][CX4]([CH3])([CH3])[CH3]",  # Boc protection
            "DPM": "[NX3][CH2][c]1[cH][cH][cH][cH][cH]1"  # Diphenylmethyl protection
        }
        
        # SMARTS patterns for deprotection (loss of protecting group)
        self.deprotection_patterns = {
            "Boc": "[NX3H2,NX3H1]",  # Free amine after Boc removal
            "DPM": "[NX3H2,NX3H1]"   # Free amine after DPM removal
        }
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        if self.sequential:
            return self._check_sequential_cycling(reactions)
        else:
            return self._check_any_cycling(reactions)
    
    def _check_sequential_cycling(self, reactions) -> Tuple[bool, int]:
        """Check if protecting groups are used sequentially as specified"""
        pg_events = []  # List of (depth, pg_type, event_type)
        
        for depth, rxn in enumerate(reactions):
            for pg in self.protection_groups:
                if self._detect_protection(rxn, pg):
                    pg_events.append((depth, pg, "protection"))
                elif self._detect_deprotection(rxn, pg):
                    pg_events.append((depth, pg, "deprotection"))
        
        # Sort events by depth (earliest first)
        pg_events.sort(key=lambda x: x[0])
        
        # Check for complete cycles: protection followed by deprotection
        cycles_found = []
        for pg in self.protection_groups:
            pg_stack = []
            for depth, pg_type, event_type in pg_events:
                if pg_type == pg:
                    if event_type == "protection":
                        pg_stack.append(depth)
                    elif event_type == "deprotection" and pg_stack:
                        protect_depth = pg_stack.pop()
                        cycles_found.append((pg, protect_depth, depth))
        
        # Check if we have cycles for all specified protecting groups
        found_pgs = set(cycle[0] for cycle in cycles_found)
        condition_met = len(found_pgs) >= len(self.protection_groups)
        
        return condition_met, len(reactions)
    
    def _check_any_cycling(self, reactions) -> Tuple[bool, int]:
        """Check if any of the specified protecting groups show cycling"""
        for pg in self.protection_groups:
            protection_found = False
            deprotection_found = False
            
            for rxn in reactions:
                if self._detect_protection(rxn, pg):
                    protection_found = True
                elif self._detect_deprotection(rxn, pg):
                    deprotection_found = True
            
            if protection_found and deprotection_found:
                return True, len(reactions)
        
        return False, len(reactions)
    
    def _detect_protection(self, rxn, pg_type):
        """Detect installation of a protecting group"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = [Chem.MolFromSmiles(s) for s in rxn_parts[0].split(".") if s]
        products = [Chem.MolFromSmiles(s) for s in rxn_parts[1].split(".") if s]
        
        if not all(reactants + products):
            return False
        
        # Count protecting group motifs in reactants vs products
        reactant_pg_count = sum(self._count_pg_motifs(mol, pg_type) for mol in reactants)
        product_pg_count = sum(self._count_pg_motifs(mol, pg_type) for mol in products)
        
        return product_pg_count > reactant_pg_count
    
    def _detect_deprotection(self, rxn, pg_type):
        """Detect removal of a protecting group"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = [Chem.MolFromSmiles(s) for s in rxn_parts[0].split(".") if s]
        products = [Chem.MolFromSmiles(s) for s in rxn_parts[1].split(".") if s]
        
        if not all(reactants + products):
            return False
        
        # Count protecting group motifs in reactants vs products
        reactant_pg_count = sum(self._count_pg_motifs(mol, pg_type) for mol in reactants)
        product_pg_count = sum(self._count_pg_motifs(mol, pg_type) for mol in products)
        
        return reactant_pg_count > product_pg_count
    
    def _count_pg_motifs(self, mol, pg_type):
        """Count the number of protecting group motifs in a molecule"""
        if mol is None or pg_type not in self.pg_patterns:
            return 0
        
        pattern = Chem.MolFromSmarts(self.pg_patterns[pg_type])
        if pattern is None:
            return 0
        
        return len(mol.GetSubstructMatches(pattern))
