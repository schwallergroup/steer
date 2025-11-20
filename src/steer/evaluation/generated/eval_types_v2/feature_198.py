"""Generated evaluation code for: Protecting group cycling strategy employed"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ProtectingGroupStrategy(MultiRxnCondBase):
    """
    Evaluates whether a specific protecting group cycling strategy is employed in the synthesis route.
    Checks for a sequence of protection/deprotection reactions on a target functional group.
    """
    
    def __init__(self, config):
        self.sequence = config["sequence"]
        self.target_group = config["target_group"]
        self.steps_involved = config["steps_involved"]
        
        # Define SMARTS patterns for different protecting groups and reactions
        self.patterns = {
            "Boc_deprotection": {
                "reactant": "[NH1,NH2][C](=O)OC(C)(C)C",  # Boc-protected amine
                "product": "[NH2,NH3+]"  # Free amine
            },
            "Cbz_protection": {
                "reactant": "[NH2,NH3+]",  # Free amine
                "product": "[NH1,NH2][C](=O)OCc1ccccc1"  # Cbz-protected amine
            },
            "Cbz_deprotection": {
                "reactant": "[NH1,NH2][C](=O)OCc1ccccc1",  # Cbz-protected amine
                "product": "[NH2,NH3+]"  # Free amine
            }
        }
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        sequence_found = self.detect_protecting_group_sequence(reactions)
        return sequence_found, len(reactions)
    
    def route_scoring(self, x):
        if x < 0:
            return 0  # Sequence not found
        else:
            # Earlier implementation of the strategy is better
            return max(0, 10 * (1 - x))
    
    def detect_protecting_group_sequence(self, reactions):
        """
        Check if the specified sequence of protecting group operations occurs in order
        """
        sequence_matches = []
        
        for rxn in reactions:
            for step in self.sequence:
                if self.detect_protection_step(rxn, step):
                    sequence_matches.append(step)
                    break
        
        # Check if we found the complete sequence in order
        return self.is_sequence_complete(sequence_matches)
    
    def detect_protection_step(self, rxn, step_type):
        """
        Detect if a reaction corresponds to a specific protection/deprotection step
        """
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants_smiles = rxn_parts[0]
            products_smiles = rxn_parts[1]
            
            # Parse reactants and products
            reactant_mols = []
            for smi in reactants_smiles.split("."):
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    reactant_mols.append(mol)
            
            product_mols = []
            for smi in products_smiles.split("."):
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    product_mols.append(mol)
            
            if not reactant_mols or not product_mols:
                return False
            
            # Check if this reaction matches the expected transformation
            pattern_info = self.patterns.get(step_type, {})
            reactant_pattern = pattern_info.get("reactant")
            product_pattern = pattern_info.get("product")
            
            if not reactant_pattern or not product_pattern:
                return False
            
            reactant_smarts = Chem.MolFromSmarts(reactant_pattern)
            product_smarts = Chem.MolFromSmarts(product_pattern)
            
            # Check if reactants contain the expected starting pattern
            reactant_match = any(mol.HasSubstructMatch(reactant_smarts) for mol in reactant_mols)
            
            # Check if products contain the expected ending pattern
            product_match = any(mol.HasSubstructMatch(product_smarts) for mol in product_mols)
            
            return reactant_match and product_match
            
        except:
            return False
    
    def is_sequence_complete(self, found_sequence):
        """
        Check if the found sequence matches the expected sequence
        """
        if len(found_sequence) < len(self.sequence):
            return False
        
        # Look for the target sequence as a subsequence in the found sequence
        target_len = len(self.sequence)
        for i in range(len(found_sequence) - target_len + 1):
            if found_sequence[i:i + target_len] == self.sequence:
                return True
        
        return False
