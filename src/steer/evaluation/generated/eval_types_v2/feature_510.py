"""Generated evaluation code for: Sequential protecting group cycling on piperidine nitrogen"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SequentialProtectingGroupCycling(MultiRxnCondBase):
    """
    Evaluates synthesis routes for sequential protecting group cycling on piperidine nitrogen.
    Checks for the presence of protection/deprotection cycles with specific protecting groups
    (benzyl and Boc) on the same nitrogen center.
    """
    
    def __init__(self, config):
        self.protection_cycles = config.get("protection_cycles", 2)
        self.protecting_groups = config.get("protecting_groups", ["benzyl", "Boc"])
        self.functional_group = config.get("functional_group", "piperidine_nitrogen")
        
        # Define SMARTS patterns for protecting groups
        self.protecting_group_patterns = {
            "benzyl": "[N;R1:1]-[CH2]-c1ccccc1",  # Benzyl-protected nitrogen
            "Boc": "[N;R1:1]-C(=O)-O-C(C)(C)C",   # Boc-protected nitrogen
            "free_piperidine": "[NH;R1:1]"         # Free piperidine nitrogen
        }

    def condition_depth(self, d):
        """
        Check if sequential protecting group cycling occurs in the synthesis route.
        Returns (condition_met, total_reactions)
        """
        reactions = self.get_rxns(d)
        
        # Track protection state changes for piperidine nitrogens
        protection_events = []
        
        for i, rxn in enumerate(reactions):
            rxn_smiles = rxn.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles:
                continue
                
            protection_change = self.detect_protection_change(rxn_smiles)
            if protection_change:
                protection_events.append((i, protection_change))
        
        # Check if we have the required cycling pattern
        condition_met = self.has_sequential_cycling(protection_events)
        
        return condition_met, len(reactions)

    def detect_protection_change(self, rxn_smiles):
        """
        Detect protection/deprotection events in a single reaction.
        Returns the type of change: 'protect_benzyl', 'deprotect_benzyl', 'protect_boc', 'deprotect_boc'
        """
        if ">>" not in rxn_smiles:
            return None
            
        reactants_smiles, products_smiles = rxn_smiles.split(">>")
        
        try:
            # Parse reactants and products
            reactant_mols = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            product_mols = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            reactant_mols = [mol for mol in reactant_mols if mol is not None]
            product_mols = [mol for mol in product_mols if mol is not None]
            
            # Count protecting groups in reactants and products
            reactant_counts = self.count_protecting_groups(reactant_mols)
            product_counts = self.count_protecting_groups(product_mols)
            
            # Determine the type of protection change
            for pg_type in self.protecting_groups:
                pg_key = pg_type.lower()
                if reactant_counts.get(pg_key, 0) < product_counts.get(pg_key, 0):
                    return f"protect_{pg_key}"
                elif reactant_counts.get(pg_key, 0) > product_counts.get(pg_key, 0):
                    return f"deprotect_{pg_key}"
                    
        except Exception:
            return None
            
        return None

    def count_protecting_groups(self, mol_list):
        """
        Count occurrences of each protecting group type in a list of molecules.
        """
        counts = {}
        
        for pg_type in self.protecting_groups:
            pg_key = pg_type.lower()
            pattern = self.protecting_group_patterns.get(pg_key)
            if pattern:
                pattern_mol = Chem.MolFromSmarts(pattern)
                if pattern_mol:
                    total_matches = 0
                    for mol in mol_list:
                        if mol:
                            matches = mol.GetSubstructMatches(pattern_mol)
                            total_matches += len(matches)
                    counts[pg_key] = total_matches
                    
        return counts

    def has_sequential_cycling(self, protection_events):
        """
        Check if the protection events show the required sequential cycling pattern.
        For benzyl/Boc cycling: protect_benzyl -> deprotect_benzyl -> protect_boc
        """
        if len(protection_events) < self.protection_cycles * 2:
            return False
            
        event_sequence = [event[1] for event in protection_events]
        
        # Look for the specific cycling pattern
        required_patterns = [
            ["protect_benzyl", "deprotect_benzyl", "protect_boc"],
            ["protect_boc", "deprotect_boc", "protect_benzyl"]
        ]
        
        for pattern in required_patterns:
            if self.contains_subsequence(event_sequence, pattern):
                return True
                
        return False

    def contains_subsequence(self, sequence, subsequence):
        """
        Check if subsequence exists within sequence (not necessarily consecutive).
        """
        if not subsequence:
            return True
            
        subseq_idx = 0
        for item in sequence:
            if subseq_idx < len(subsequence) and item == subsequence[subseq_idx]:
                subseq_idx += 1
                if subseq_idx == len(subsequence):
                    return True
                    
        return False
