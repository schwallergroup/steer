"""Generated evaluation code for: Differential protecting group strategy for diol"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class DifferentialProtectingGroupStrategy(BaseScoring):
    """
    Evaluates synthesis routes for differential protecting group strategies on diols.
    Checks if different protecting groups (TBS vs acetate) are used to selectively
    protect primary vs secondary alcohols in diol substrates.
    """
    
    def __init__(self, config: Dict):
        self.substrate_pattern = config["parameters"]["substrate_pattern"]
        self.protecting_groups = config["parameters"]["protecting_groups"]
        self.selectivity = config["parameters"]["selectivity"]
        
        # Define SMARTS patterns for protecting groups
        self.pg_patterns = {
            "TBS": "[CH3][Si]([CH3])([CH3])[CH2][CH3]",  # tert-butyldimethylsilyl
            "acetate": "[CH3]C(=O)O",  # acetyl group
        }
        
        # Pattern for diol substrate
        self.diol_pattern = Chem.MolFromSmarts(self.substrate_pattern)
    
    def route_scoring(self, x) -> float:
        """Convert depth to score - earlier use of differential protection is better"""
        if x < 0:
            return 0  # Strategy not found
        else:
            return 1 - x  # Earlier application scores higher
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction applies differential protecting groups to a diol"""
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            rxn_parts = rxn_smiles.split(">>")
            
            if len(rxn_parts) != 2:
                return False
                
            reactants = rxn_parts[0]
            products = rxn_parts[1]
            
            # Parse reactants and products
            reactant_mols = [Chem.MolFromSmiles(smi.strip()) for smi in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(smi.strip()) for smi in products.split(".")]
            
            # Filter out None molecules
            reactant_mols = [mol for mol in reactant_mols if mol is not None]
            product_mols = [mol for mol in product_mols if mol is not None]
            
            # Check if any reactant contains the diol pattern
            has_diol_substrate = any(mol.HasSubstructMatch(self.diol_pattern) for mol in reactant_mols)
            
            if not has_diol_substrate:
                return False
            
            # Check if products contain both protecting groups
            pg_found = set()
            for mol in product_mols:
                for pg_name, pg_pattern in self.pg_patterns.items():
                    if pg_name in self.protecting_groups:
                        pg_smarts = Chem.MolFromSmarts(pg_pattern)
                        if pg_smarts and mol.HasSubstructMatch(pg_smarts):
                            pg_found.add(pg_name)
            
            # For differential selectivity, we need both protecting groups present
            if self.selectivity == "differential":
                expected_pgs = set(self.protecting_groups)
                return len(pg_found.intersection(expected_pgs)) >= 2
            
            # Check if at least one expected protecting group is present
            return len(pg_found.intersection(set(self.protecting_groups))) > 0
            
        except Exception:
            return False
    
    def _has_protecting_group_reagent(self, reactant_mols) -> bool:
        """Check if protecting group reagents are present in reactants"""
        reagent_patterns = {
            "TBS": "[Si]([CH3])([CH3])[CH2][CH3]",  # TBS reagents
            "acetate": "CC(=O)Cl",  # acetyl chloride or similar
        }
        
        for mol in reactant_mols:
            for pg_name in self.protecting_groups:
                if pg_name in reagent_patterns:
                    pattern = Chem.MolFromSmarts(reagent_patterns[pg_name])
                    if pattern and mol.HasSubstructMatch(pattern):
                        return True
        return False
