"""Generated evaluation code for: Multiple protecting group swaps on nitrogen"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ProtectingGroupSwaps(MultiRxnCondBase):
    """
    Evaluates routes based on multiple protecting group swaps on nitrogen atoms.
    Checks if the route performs a specific sequence of protecting group changes.
    """
    
    def __init__(self, config):
        self.atom_type = config["atom_type"]
        self.required_swap_count = config["swap_count"]
        self.target_sequence = config["groups"]
        
        # Define protecting group SMARTS patterns
        self.pg_patterns = {
            "Boc": "[NX3][CX3](=[OX1])[OX2][CX4]([CH3])([CH3])[CH3]",  # tert-butoxycarbonyl
            "Benzyl": "[NX3][CH2][cX3]1[cX3H][cX3H][cX3H][cX3H][cX3H]1",  # benzyl
            "H": "[NX3H]",  # free amine
            "Cbz": "[NX3][CX3](=[OX1])[OX2][CH2][cX3]1[cX3H][cX3H][cX3H][cX3H][cX3H]1",  # benzyloxycarbonyl
            "Fmoc": "[NX3][CX3](=[OX1])[OX2][CH2][CH]1c2ccccc2c3c1cccc3",  # fluorenylmethyloxycarbonyl
            "Tosyl": "[NX3][SX4](=[OX1])(=[OX1])[cX3]1[cX3H][cX3H][cX3]([CH3])[cX3H][cX3H]1"  # tosyl
        }
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Track protecting group changes throughout the route
        pg_sequence = self.extract_pg_sequence(reactions)
        
        # Check if we have the required number of swaps
        swap_count = len(pg_sequence) - 1 if pg_sequence else 0
        
        # Check if sequence matches target
        sequence_matches = self.sequence_matches_target(pg_sequence)
        
        condition = (swap_count >= self.required_swap_count and sequence_matches)
        return condition, len(reactions)
    
    def extract_pg_sequence(self, reactions):
        """Extract the sequence of protecting groups on nitrogen through the route."""
        sequence = []
        
        for rxn in reactions:
            # Parse reaction SMILES
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                continue
                
            reactants = rxn_parts[0]
            products = rxn_parts[1].split(".")
            
            # Check for protecting group changes
            reactant_mol = Chem.MolFromSmiles(reactants)
            if not reactant_mol:
                continue
                
            for product_smiles in products:
                product_mol = Chem.MolFromSmiles(product_smiles)
                if not product_mol:
                    continue
                    
                # Detect protecting group on nitrogen in reactant and product
                reactant_pg = self.detect_nitrogen_pg(reactant_mol)
                product_pg = self.detect_nitrogen_pg(product_mol)
                
                # If there's a change, record it
                if reactant_pg != product_pg and product_pg is not None:
                    if not sequence:  # First entry
                        sequence.append(reactant_pg)
                    sequence.append(product_pg)
                    
        return sequence
    
    def detect_nitrogen_pg(self, mol):
        """Detect which protecting group is on nitrogen atoms in the molecule."""
        if not mol:
            return None
            
        # Check each protecting group pattern
        for pg_name, pattern in self.pg_patterns.items():
            if pattern and mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                return pg_name
                
        # Check if nitrogen is present but unprotected
        free_amine_pattern = Chem.MolFromSmarts("[NX3H2,NX3H1]")
        if mol.HasSubstructMatch(free_amine_pattern):
            return "H"
            
        return None
    
    def sequence_matches_target(self, observed_sequence):
        """Check if the observed protecting group sequence matches the target."""
        if len(observed_sequence) != len(self.target_sequence):
            return False
            
        # Allow for partial matches if observed sequence is longer
        for i, target_pg in enumerate(self.target_sequence):
            if i >= len(observed_sequence):
                return False
            if observed_sequence[i] != target_pg:
                return False
                
        return True
    
    def route_scoring(self, x):
        """Score the route based on protecting group swap strategy."""
        if x < 0:
            return 0  # Strategy not found
        else:
            # Better score for routes that implement the strategy earlier
            return max(0, 1 - x)
