"""Generated evaluation code for: Multiple protecting group cycling strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ProtectingGroupCycling(MultiRxnCondBase):
    """
    Evaluates synthesis routes based on protecting group cycling strategy.
    Checks if a specified atom type undergoes multiple protecting group changes
    throughout the synthesis route.
    """
    
    def __init__(self, config):
        self.atom_type = config["atom_type"].lower()
        self.min_cycles = config["min_cycles"]
        self.protection_changes = [pg.upper() for pg in config["protection_changes"]]
        
        # Define protecting group SMARTS patterns
        self.pg_patterns = {
            "TFA": "[NH1]C(=O)C(F)(F)F",  # Trifluoroacetyl
            "CBZ": "[NH1]C(=O)Oc1ccccc1",  # Carbobenzyloxy
            "BOC": "[NH1]C(=O)OC(C)(C)C",  # tert-Butoxycarbonyl
            "BENZYL": "[NH1]Cc1ccccc1",    # Benzyl
            "FMOC": "[NH1]C(=O)OCc1ccccc1c2ccccc12"  # Fluorenylmethoxycarbonyl
        }
    
    def condition_depth(self, d):
        """Check if the route contains sufficient protecting group cycling."""
        reactions = self.get_rxns(d)
        pg_changes = self.count_protecting_group_changes(reactions)
        
        condition = pg_changes >= self.min_cycles
        return condition, len(reactions)
    
    def count_protecting_group_changes(self, reactions):
        """Count protecting group addition/removal cycles throughout the route."""
        pg_changes = 0
        
        for rxn_smiles in reactions:
            if self.detect_protecting_group_change(rxn_smiles):
                pg_changes += 1
                
        return pg_changes
    
    def detect_protecting_group_change(self, rxn_smiles):
        """Detect if a reaction involves protecting group addition or removal."""
        try:
            rxn_parts = rxn_smiles.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants = rxn_parts[0]
            products = rxn_parts[1]
            
            reactant_mols = []
            for r_smi in reactants.split("."):
                mol = Chem.MolFromSmiles(r_smi)
                if mol:
                    reactant_mols.append(mol)
            
            product_mols = []
            for p_smi in products.split("."):
                mol = Chem.MolFromSmiles(p_smi)
                if mol:
                    product_mols.append(mol)
            
            # Check for protecting group addition (PG appears in products but not reactants)
            if self.is_pg_addition(reactant_mols, product_mols):
                return True
                
            # Check for protecting group removal (PG appears in reactants but not products)
            if self.is_pg_removal(reactant_mols, product_mols):
                return True
                
            return False
            
        except Exception:
            return False
    
    def is_pg_addition(self, reactants, products):
        """Check if protecting group is added in this reaction."""
        reactant_pg_count = self.count_pg_matches(reactants)
        product_pg_count = self.count_pg_matches(products)
        
        # Also check for free atom type that gets protected
        reactant_free_atoms = self.count_free_atoms(reactants)
        product_free_atoms = self.count_free_atoms(products)
        
        return (product_pg_count > reactant_pg_count) and (product_free_atoms < reactant_free_atoms)
    
    def is_pg_removal(self, reactants, products):
        """Check if protecting group is removed in this reaction."""
        reactant_pg_count = self.count_pg_matches(reactants)
        product_pg_count = self.count_pg_matches(products)
        
        # Also check for free atom type that gets deprotected
        reactant_free_atoms = self.count_free_atoms(reactants)
        product_free_atoms = self.count_free_atoms(products)
        
        return (reactant_pg_count > product_pg_count) and (reactant_free_atoms < product_free_atoms)
    
    def count_pg_matches(self, mols):
        """Count protecting group matches across all molecules."""
        total_matches = 0
        
        for pg_name in self.protection_changes:
            if pg_name in self.pg_patterns:
                pattern = Chem.MolFromSmarts(self.pg_patterns[pg_name])
                if pattern:
                    for mol in mols:
                        if mol and mol.HasSubstructMatch(pattern):
                            total_matches += len(mol.GetSubstructMatches(pattern))
                            
        return total_matches
    
    def count_free_atoms(self, mols):
        """Count free atoms of the specified type (e.g., free NH groups)."""
        if self.atom_type == "nitrogen":
            free_pattern = Chem.MolFromSmarts("[NH2,NH1][!C(=O)]")  # Free NH not bound to carbonyl
        elif self.atom_type == "oxygen":
            free_pattern = Chem.MolFromSmarts("[OH1][!S,!P]")  # Free OH not on sulfur/phosphorus
        else:
            return 0
            
        if not free_pattern:
            return 0
            
        total_free = 0
        for mol in mols:
            if mol and mol.HasSubstructMatch(free_pattern):
                total_free += len(mol.GetSubstructMatches(free_pattern))
                
        return total_free
    
    def route_scoring(self, x):
        """Convert condition result to score (0-10 scale)."""
        if x < 0:
            return 0  # Condition not met
        else:
            # Better score for meeting the minimum cycles requirement
            return 10.0 - min(10.0, x * 2.0)  # Penalize excess complexity slightly
