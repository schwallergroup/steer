"""Generated evaluation code for: Orthogonal protecting group strategy for phenol differentiation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class OrthogonalProtectingGroupStrategy(MultiRxnCondBase):
    """
    Evaluates whether an orthogonal protecting group strategy is used for phenol differentiation.
    Checks for the presence of both acetate and benzyl protecting groups in the synthesis route.
    """
    
    def __init__(self, config):
        self.protecting_groups = config.get("protecting_groups", ["acetate", "benzyl"])
        self.strategy_type = config.get("strategy_type", "orthogonal")
        self.functional_group = config.get("functional_group", "phenol")
        
        # Define SMARTS patterns for protecting groups
        self.pg_patterns = {
            "acetate": "[OH1][C](=O)[CH3]",  # Acetate ester
            "benzyl": "[OH1][CH2]c1ccccc1",   # Benzyl ether
        }
        
        # Phenol pattern for detection
        self.phenol_pattern = "c[OH1]"
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Track which protecting groups are introduced
        pg_introduced = set()
        
        for rxn in reactions:
            for pg_name in self.protecting_groups:
                if self.detect_protecting_group_introduction(rxn, pg_name):
                    pg_introduced.add(pg_name)
        
        # Check if orthogonal strategy is achieved
        if self.strategy_type == "orthogonal":
            # All specified protecting groups should be present
            condition = len(pg_introduced) >= len(self.protecting_groups)
        else:
            # At least one protecting group should be present
            condition = len(pg_introduced) > 0
            
        return condition, len(reactions)
    
    def detect_protecting_group_introduction(self, rxn, pg_name):
        """
        Detects if a specific protecting group is introduced in a reaction.
        """
        if pg_name not in self.pg_patterns:
            return False
            
        pattern = self.pg_patterns[pg_name]
        rxn_parts = rxn.split(">>")
        
        if len(rxn_parts) != 2:
            return False
            
        reactants_smiles = rxn_parts[0]
        products_smiles = rxn_parts[1]
        
        try:
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
            
            # Count protecting group occurrences
            pg_mol = Chem.MolFromSmarts(pattern)
            if not pg_mol:
                return False
                
            reactant_pg_count = sum(len(mol.GetSubstructMatches(pg_mol)) for mol in reactant_mols)
            product_pg_count = sum(len(mol.GetSubstructMatches(pg_mol)) for mol in product_mols)
            
            # Also check for free phenols being protected
            phenol_mol = Chem.MolFromSmarts(self.phenol_pattern)
            if phenol_mol:
                reactant_phenol_count = sum(len(mol.GetSubstructMatches(phenol_mol)) for mol in reactant_mols)
                product_phenol_count = sum(len(mol.GetSubstructMatches(phenol_mol)) for mol in product_mols)
                
                # Protection reaction: phenol decreases, protecting group increases
                return (product_pg_count > reactant_pg_count) and (product_phenol_count < reactant_phenol_count)
            
            return product_pg_count > reactant_pg_count
            
        except Exception:
            return False
