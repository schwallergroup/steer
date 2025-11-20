"""Generated evaluation code for: Minimal protecting group strategy with unprotected nucleophiles"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class MinimalProtectingGroupStrategy(MultiRxnCondBase):
    """
    Evaluates routes based on minimal protecting group strategy.
    Checks that specified reactive groups remain unprotected throughout the route
    while avoiding protection/deprotection reactions.
    """
    
    def __init__(self, config):
        self.protected_groups = config.get("protected_groups", [])
        self.unprotected_reactive_groups = config.get("unprotected_reactive_groups", [])
        self.strategy = config.get("strategy", "minimal")
        
        # Define SMARTS patterns for reactive groups
        self.group_patterns = {
            "amine": ["[NH2]", "[NH1]"],  # Primary and secondary amines
            "phenol": ["c[OH]"],  # Phenolic OH
            "alcohol": ["[CH2,CH][OH]"],  # Aliphatic alcohols
            "carboxylic_acid": ["C(=O)[OH]"],
            "thiol": ["[SH]"]
        }
        
        # Define SMARTS patterns for common protecting groups
        self.protection_patterns = {
            "boc": ["NC(=O)OC(C)(C)C"],  # Boc-protected amine
            "cbz": ["NC(=O)OCc1ccccc1"],  # Cbz-protected amine
            "fmoc": ["NC(=O)OCC1c2ccccc2-c2ccccc21"],  # Fmoc-protected amine
            "tbdms": ["[OH,NH]S(=O)(=O)C(C)(C)C"],  # TBDMS-protected OH/NH
            "acetyl": ["NC(=O)C", "OC(=O)C"],  # Acetyl protection
            "benzyl": ["OCc1ccccc1"]  # Benzyl protection
        }

    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Check for protection/deprotection reactions
        has_protection_reactions = any(self.is_protection_reaction(r) for r in reactions)
        
        # Check that unprotected groups remain unprotected
        maintains_unprotected = self.check_unprotected_groups_maintained(reactions)
        
        # Check that protected groups are properly protected
        avoids_forbidden_protection = not any(
            self.has_forbidden_protected_group(r) for r in reactions
        )
        
        if self.strategy == "minimal":
            condition = (not has_protection_reactions and 
                        maintains_unprotected and 
                        avoids_forbidden_protection)
        else:
            condition = maintains_unprotected and avoids_forbidden_protection
            
        return condition, len(reactions)

    def is_protection_reaction(self, rxn):
        """Detect protection or deprotection reactions"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        try:
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".")]
            
            if None in reactant_mols or None in product_mols:
                return False
            
            # Check for protecting group reagents or byproducts
            protecting_reagents = [
                "CC(C)(C)OC(=O)Cl",  # Boc-Cl
                "ClC(=O)OCc1ccccc1",  # Cbz-Cl
                "CC(=O)Cl",  # Acetyl chloride
                "c1ccc(COCl)cc1",  # Benzyl chloride
                "CC(C)(C)[Si](Cl)(C)C"  # TBDMS-Cl
            ]
            
            # Check if any protecting group reagents are present
            reactant_smiles = [Chem.MolToSmiles(mol) for mol in reactant_mols]
            for reagent in protecting_reagents:
                if any(reagent in r_smi for r_smi in reactant_smiles):
                    return True
                    
            # Check for protection/deprotection by comparing functional groups
            reactant_groups = self.count_functional_groups(reactant_mols)
            product_groups = self.count_functional_groups(product_mols)
            
            # If free functional groups decrease, likely protection
            # If free functional groups increase, likely deprotection
            for group in self.unprotected_reactive_groups:
                if group in reactant_groups and group in product_groups:
                    if reactant_groups[group] != product_groups[group]:
                        return True
                        
            return False
            
        except:
            return False

    def check_unprotected_groups_maintained(self, reactions):
        """Check that specified groups remain unprotected throughout"""
        for rxn in reactions:
            try:
                rxn_parts = rxn.split(">>")
                if len(rxn_parts) != 2:
                    continue
                    
                products = rxn_parts[1]
                product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".")]
                
                if None in product_mols:
                    continue
                
                # Check that unprotected groups are still present
                for group in self.unprotected_reactive_groups:
                    if not self.has_unprotected_group(product_mols, group):
                        # Allow if the group was legitimately consumed in reaction
                        if not self.group_consumed_in_reaction(rxn, group):
                            return False
                            
            except:
                continue
                
        return True

    def has_forbidden_protected_group(self, rxn):
        """Check if any forbidden protected groups are present"""
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            products = rxn_parts[1]
            product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".")]
            
            if None in product_mols:
                return False
                
            for mol in product_mols:
                for group in self.protected_groups:
                    if group in self.protection_patterns:
                        patterns = self.protection_patterns[group]
                        if isinstance(patterns, str):
                            patterns = [patterns]
                        for pattern in patterns:
                            if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                                return True
                                
            return False
            
        except:
            return False

    def has_unprotected_group(self, mols, group_type):
        """Check if unprotected version of group is present"""
        if group_type not in self.group_patterns:
            return True
            
        patterns = self.group_patterns[group_type]
        if isinstance(patterns, str):
            patterns = [patterns]
            
        for mol in mols:
            for pattern in patterns:
                try:
                    if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                        return True
                except:
                    continue
                    
        return False

    def count_functional_groups(self, mols):
        """Count functional groups in molecules"""
        counts = {}
        for group_type, patterns in self.group_patterns.items():
            if isinstance(patterns, str):
                patterns = [patterns]
            counts[group_type] = 0
            for mol in mols:
                for pattern in patterns:
                    try:
                        matches = mol.GetSubstructMatches(Chem.MolFromS
