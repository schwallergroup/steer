"""Generated evaluation code for: Sequential protecting group swap strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ProtectingGroupSequence(MultiRxnCondBase):
    """
    Evaluates whether a synthesis route follows a sequential protecting group swap strategy.
    Checks if the route contains the specified sequence of protecting group changes
    with the expected number of swaps.
    """
    
    def __init__(self, config):
        self.protection_sequence = config["protection_sequence"]
        self.expected_swap_count = config["swap_count"]
        
        # Define SMARTS patterns for common protecting groups
        self.protecting_group_patterns = {
            "benzyl": "[NH1,NH2][CH2]c1ccccc1",  # N-benzyl
            "boc": "[NH1,NH2]C(=O)OC(C)(C)C",   # N-Boc
            "cbz": "[NH1,NH2]C(=O)O[CH2]c1ccccc1",  # N-Cbz
            "fmoc": "[NH1,NH2]C(=O)OCC1c2ccccc2-c2ccccc21",  # N-Fmoc
            "tosyl": "[NH1,NH2]S(=O)(=O)c1ccc(C)cc1",  # N-tosyl
            "acetyl": "[NH1,NH2]C(=O)C"  # N-acetyl
        }
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Track protecting group changes throughout the route
        pg_sequence = self.extract_protection_sequence(reactions)
        
        # Check if the observed sequence matches the expected pattern
        condition_met = self.matches_expected_sequence(pg_sequence)
        
        return condition_met, len(reactions)
    
    def extract_protection_sequence(self, reactions) -> List[str]:
        """Extract the sequence of protecting group changes from reactions."""
        sequence = []
        
        for rxn in reactions:
            pg_change = self.detect_protecting_group_change(rxn)
            if pg_change:
                sequence.append(pg_change)
        
        return sequence
    
    def detect_protecting_group_change(self, rxn) -> str:
        """
        Detect if a reaction involves protecting group installation/removal.
        Returns the type of change (e.g., 'benzyl_to_boc', 'boc_removal', etc.)
        """
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return None
                
            reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[0].split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in rxn_parts[1].split(".")]
            
            if not all(reactants + products):
                return None
            
            # Find protecting groups in reactants and products
            reactant_pgs = set()
            product_pgs = set()
            
            for mol in reactants:
                reactant_pgs.update(self.find_protecting_groups(mol))
            
            for mol in products:
                product_pgs.update(self.find_protecting_groups(mol))
            
            # Determine the type of change
            removed_pgs = reactant_pgs - product_pgs
            added_pgs = product_pgs - reactant_pgs
            
            if len(removed_pgs) == 1 and len(added_pgs) == 1:
                removed_pg = list(removed_pgs)[0]
                added_pg = list(added_pgs)[0]
                return f"{removed_pg}_to_{added_pg}"
            elif len(removed_pgs) == 1 and len(added_pgs) == 0:
                return f"{list(removed_pgs)[0]}_removal"
            elif len(removed_pgs) == 0 and len(added_pgs) == 1:
                return f"{list(added_pgs)[0]}_installation"
                
        except Exception:
            pass
        
        return None
    
    def find_protecting_groups(self, mol) -> Set[str]:
        """Find all protecting groups present in a molecule."""
        found_pgs = set()
        
        if mol is None:
            return found_pgs
        
        for pg_name, pattern in self.protecting_group_patterns.items():
            try:
                patt_mol = Chem.MolFromSmarts(pattern)
                if patt_mol and mol.HasSubstructMatch(patt_mol):
                    found_pgs.add(pg_name)
            except Exception:
                continue
        
        return found_pgs
    
    def matches_expected_sequence(self, observed_sequence: List[str]) -> bool:
        """
        Check if the observed protecting group sequence matches the expected pattern.
        """
        if len(observed_sequence) != self.expected_swap_count:
            return False
        
        # Build expected transition patterns
        expected_transitions = []
        for i in range(len(self.protection_sequence) - 1):
            from_pg = self.protection_sequence[i]
            to_pg = self.protection_sequence[i + 1]
            expected_transitions.append(f"{from_pg}_to_{to_pg}")
        
        # Check if we have the right number of expected transitions
        if len(expected_transitions) != self.expected_swap_count:
            return False
        
        # Match observed sequence with expected transitions
        matches = 0
        for transition in expected_transitions:
            if transition in observed_sequence:
                matches += 1
        
        # All expected transitions should be present
        return matches == self.expected_swap_count
