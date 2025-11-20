"""Generated evaluation code for: Multiple protecting group strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ProtectingGroupStrategy(MultiRxnCondBase):
    """
    Evaluates whether a synthesis route employs multiple protecting group strategies.
    Checks for the presence of specified protecting groups and validates the count matches requirements.
    """
    
    def __init__(self, config):
        self.protecting_groups = config["protecting_groups"]
        self.required_count = config["count"]
        
        # Define SMARTS patterns for common protecting groups
        self.protecting_group_patterns = {
            "DMT": "[OH]C([c]1ccc(OC)cc1)([c]2ccc(OC)cc2)[c]3ccccc3",  # Dimethoxytrityl
            "Boc": "NC(=O)OC(C)(C)C",  # tert-Butoxycarbonyl
            "Cbz": "NC(=O)OCc1ccccc1",  # Carbobenzyloxy
            "Fmoc": "NC(=O)OCC1c2ccccc2-c2ccccc12",  # Fluorenylmethyloxycarbonyl
            "TBS": "[OH,NH]S(=O)(=O)C(C)(C)C",  # tert-Butyldimethylsilyl
            "TIPS": "[OH,NH]Si(C(C)C)(C(C)C)C(C)C",  # Triisopropylsilyl
            "Ac": "NC(=O)C",  # Acetyl
            "Bn": "[OH,NH]Cc1ccccc1",  # Benzyl
        }
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        found_groups = set()
        
        # Check each reaction for protecting group operations
        for rxn in reactions:
            for group_name in self.protecting_groups:
                if group_name in self.protecting_group_patterns:
                    if self.detect_protecting_group_reaction(rxn, group_name):
                        found_groups.add(group_name)
        
        # Check if we found the required number of different protecting groups
        condition_met = len(found_groups) >= self.required_count and \
                       all(group in found_groups for group in self.protecting_groups)
        
        return condition_met, len(reactions)
    
    def detect_protecting_group_reaction(self, rxn, group_name):
        """
        Detect if a reaction involves the specified protecting group.
        Looks for either protection (group appears in product) or deprotection (group disappears).
        """
        if group_name not in self.protecting_group_patterns:
            return False
            
        pattern = self.protecting_group_patterns[group_name]
        
        try:
            # Parse reaction SMILES
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants_smiles = rxn_parts[0]
            products_smiles = rxn_parts[1]
            
            # Check reactants for protecting group
            reactant_has_group = False
            for reactant_smi in reactants_smiles.split("."):
                reactant_mol = Chem.MolFromSmiles(reactant_smi)
                if reactant_mol and self.has_protecting_group(reactant_mol, pattern):
                    reactant_has_group = True
                    break
            
            # Check products for protecting group
            product_has_group = False
            for product_smi in products_smiles.split("."):
                product_mol = Chem.MolFromSmiles(product_smi)
                if product_mol and self.has_protecting_group(product_mol, pattern):
                    product_has_group = True
                    break
            
            # Protection or deprotection occurred if group status changed
            return reactant_has_group != product_has_group
            
        except Exception:
            return False
    
    def has_protecting_group(self, mol, pattern):
        """Check if molecule contains the protecting group pattern."""
        try:
            pattern_mol = Chem.MolFromSmarts(pattern)
            if pattern_mol is None:
                return False
            return mol.HasSubstructMatch(pattern_mol)
        except Exception:
            return False
    
    def route_scoring(self, x):
        """
        Score based on whether the protecting group strategy is successfully employed.
        Returns high score (8-10) if condition is met, low score (0-2) otherwise.
        """
        if x < 0:
            return 0  # Strategy not found
        else:
            return 10  # Strategy successfully employed
