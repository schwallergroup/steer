"""Generated evaluation code for: Multiple orthogonal protecting group strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class MultipleProtectingGroupStrategy(MultiRxnCondBase):
    """
    Evaluates synthesis routes for the use of multiple orthogonal protecting group strategies.
    Checks for the presence of specified protecting groups and their corresponding deprotection methods.
    """
    
    def __init__(self, config):
        self.protecting_groups = config.get("protecting_groups", [])
        self.deprotection_methods = config.get("deprotection_methods", [])
        self.orthogonality = config.get("orthogonality", "multiple")
        
        # Define SMARTS patterns for protecting groups
        self.pg_patterns = {
            "benzyl_ether": "[CH2]c1ccccc1-O-[*]",
            "trifluoroacetamide": "FC(F)(F)C(=O)N[*]",
            "boc_carbamate": "CC(C)(C)OC(=O)N[*]"
        }
        
        # Define reaction patterns for deprotection methods
        self.deprotection_patterns = {
            "hydrogenation": "[H][H]",  # H2 as reagent
            "base": "[OH-]",  # Hydroxide base
            "acid": "[H+]"   # Acid conditions
        }
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Track which protecting groups and deprotection methods are found
        found_protection = set()
        found_deprotection = set()
        
        for rxn in reactions:
            # Check for protection reactions (forming protecting groups)
            for pg_name, pattern in self.pg_patterns.items():
                if pg_name in self.protecting_groups and self.detect_protection(rxn, pattern):
                    found_protection.add(pg_name)
            
            # Check for deprotection reactions
            for method_name in self.deprotection_methods:
                if self.detect_deprotection(rxn, method_name):
                    found_deprotection.add(method_name)
        
        # Evaluate orthogonality condition
        if self.orthogonality == "multiple":
            # Require at least 2 different protecting groups and 2 different deprotection methods
            protection_condition = len(found_protection) >= 2
            deprotection_condition = len(found_deprotection) >= 2
            condition = protection_condition and deprotection_condition
        else:
            # Require all specified protecting groups and deprotection methods
            protection_condition = len(found_protection) == len(self.protecting_groups)
            deprotection_condition = len(found_deprotection) == len(self.deprotection_methods)
            condition = protection_condition and deprotection_condition
        
        return condition, len(reactions)
    
    def detect_protection(self, rxn, pattern):
        """Detect if a protecting group is being installed in the reaction"""
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
            
            reactants = rxn_parts[0]
            products = rxn_parts[1]
            
            # Check if protecting group appears in products but not in reactants
            product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".") if p.strip()]
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".") if r.strip()]
            
            pattern_mol = Chem.MolFromSmarts(pattern)
            if pattern_mol is None:
                return False
            
            # Check if pattern exists in products
            product_has_pattern = any(mol and mol.HasSubstructMatch(pattern_mol) for mol in product_mols)
            
            # Check if pattern exists in reactants
            reactant_has_pattern = any(mol and mol.HasSubstructMatch(pattern_mol) for mol in reactant_mols)
            
            # Protection: pattern appears in products but not in reactants (or less in reactants)
            return product_has_pattern and not reactant_has_pattern
            
        except Exception:
            return False
    
    def detect_deprotection(self, rxn, method):
        """Detect if a specific deprotection method is being used"""
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
            
            reactants = rxn_parts[0]
            products = rxn_parts[1]
            
            if method == "hydrogenation":
                # Look for H2 in reactants or Pd catalyst patterns
                return "[H][H]" in reactants or "Pd" in reactants
            
            elif method == "base":
                # Look for basic conditions (hydroxide, carbonate, etc.)
                base_indicators = ["[OH-]", "[O-]", "CO3", "NaOH", "KOH", "LiOH"]
                return any(indicator in reactants for indicator in base_indicators)
            
            elif method == "acid":
                # Look for acidic conditions
                acid_indicators = ["[H+]", "HCl", "TFA", "H2SO4", "HBr", "CF3COOH"]
                return any(indicator in reactants for indicator in acid_indicators)
            
            return False
            
        except Exception:
            return False
