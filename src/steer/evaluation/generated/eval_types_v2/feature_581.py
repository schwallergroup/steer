"""Generated evaluation code for: Sequential protecting group cycling strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SequentialProtectingGroupCycling(MultiRxnCondBase):
    """
    Evaluates if a synthesis route uses sequential protecting group cycling strategy.
    Checks for sequential deprotection of one group followed by protection with another
    on the same functional group type.
    """
    
    def __init__(self, config):
        self.protecting_groups = config["parameters"]["protecting_groups"]
        self.sequence = config["parameters"]["sequence"]
        self.target_functional_group = config["parameters"]["target_functional_group"]
        
        # Define SMARTS patterns for protecting groups
        self.pg_patterns = {
            "benzyl": {
                "protected": "[NX3][CH2]c1ccccc1",  # N-benzyl amine
                "deprotected": "[NX3H2,NX3H1]"      # Free amine
            },
            "trifluoroacetyl": {
                "protected": "[NX3]C(=O)C(F)(F)F",  # N-trifluoroacetyl amine
                "deprotected": "[NX3H2,NX3H1]"      # Free amine
            }
        }
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        """Check if sequential protecting group cycling occurs in the route."""
        reactions = self.get_rxns(d)
        total_reactions = len(reactions)
        
        if total_reactions < 2:
            return False, total_reactions
        
        # Look for sequential pattern: deprotection followed by protection
        found_sequence = self.detect_sequential_cycling(reactions)
        
        return found_sequence, total_reactions
    
    def detect_sequential_cycling(self, reactions):
        """Detect if the protecting group cycling occurs sequentially."""
        if len(self.protecting_groups) != 2:
            return False
            
        pg1, pg2 = self.protecting_groups
        
        # Look for pattern: PG1-deprotection -> PG2-protection
        for i in range(len(reactions) - 1):
            # Check if reaction i is deprotection of first PG
            if self.is_deprotection(reactions[i], pg1):
                # Check if next reaction is protection with second PG
                if self.is_protection(reactions[i + 1], pg2):
                    return True
        
        return False
    
    def is_deprotection(self, rxn, protecting_group):
        """Check if reaction removes specified protecting group."""
        try:
            reactants, products = self.parse_reaction(rxn)
            
            if not reactants or not products:
                return False
            
            # Check if reactants have the protecting group and products don't
            reactant_has_pg = any(self.mol_has_protecting_group(mol, protecting_group, "protected") 
                                for mol in reactants)
            product_lacks_pg = any(self.mol_has_protecting_group(mol, protecting_group, "deprotected") 
                                 for mol in products)
            
            return reactant_has_pg and product_lacks_pg
            
        except Exception:
            return False
    
    def is_protection(self, rxn, protecting_group):
        """Check if reaction adds specified protecting group."""
        try:
            reactants, products = self.parse_reaction(rxn)
            
            if not reactants or not products:
                return False
            
            # Check if reactants lack the protecting group and products have it
            reactant_lacks_pg = any(self.mol_has_protecting_group(mol, protecting_group, "deprotected") 
                                  for mol in reactants)
            product_has_pg = any(self.mol_has_protecting_group(mol, protecting_group, "protected") 
                               for mol in products)
            
            return reactant_lacks_pg and product_has_pg
            
        except Exception:
            return False
    
    def mol_has_protecting_group(self, mol, protecting_group, state):
        """Check if molecule has protecting group in specified state."""
        if protecting_group not in self.pg_patterns:
            return False
            
        pattern_smarts = self.pg_patterns[protecting_group][state]
        pattern = Chem.MolFromSmarts(pattern_smarts)
        
        if pattern is None:
            return False
            
        return mol.HasSubstructMatch(pattern)
    
    def parse_reaction(self, rxn_smiles):
        """Parse reaction SMILES into reactant and product molecules."""
        try:
            parts = rxn_smiles.split(">>")
            if len(parts) != 2:
                return None, None
                
            reactant_smiles = parts[0].split(".")
            product_smiles = parts[1].split(".")
            
            reactants = [Chem.MolFromSmiles(smi) for smi in reactant_smiles if smi]
            products = [Chem.MolFromSmiles(smi) for smi in product_smiles if smi]
            
            # Filter out None molecules
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            return reactants, products
            
        except Exception:
            return None, None
