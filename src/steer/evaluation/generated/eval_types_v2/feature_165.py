"""Generated evaluation code for: Late stage Cbz deprotection"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageProtectingGroupStrategy(BaseScoring):
    """
    Evaluates whether a specific protecting group operation (protection/deprotection) 
    occurs at the desired timing in the synthesis route.
    """
    
    def __init__(self, config: Dict):
        self.protecting_group = config["parameters"]["protecting_group"]
        self.operation = config["parameters"]["operation"]  # "protection" or "deprotection"
        self.timing = config["parameters"]["timing"]  # "early" or "late"
        
        # Define SMARTS patterns for common protecting groups
        self.pg_patterns = {
            "Cbz": "[NH1,NH0]-C(=O)-O-CH2-c1ccccc1",  # Carboxybenzyl
            "Boc": "[NH1,NH0]-C(=O)-O-C(C)(C)C",       # tert-Butoxycarbonyl
            "Fmoc": "[NH1,NH0]-C(=O)-O-CH2-C1-c2ccccc2-c3ccccc3-1",  # Fluorenylmethyloxycarbonyl
            "Ac": "[NH1,NH0]-C(=O)-CH3",               # Acetyl
            "Bn": "[NH1,NH0]-CH2-c1ccccc1",            # Benzyl
        }
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Operation doesn't occur
        
        if self.timing == "late":
            return 1 - x  # Later is better, score decreases with earlier timing
        elif self.timing == "early":
            return x  # Earlier is better, score increases with later timing
        else:
            return 0
    
    def hit_condition(self, d) -> bool:
        """
        Check if the current reaction involves the specified protecting group operation.
        """
        if "mapped_reaction_smiles" not in d.get("metadata", {}):
            return False
            
        rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
        rxn_parts = rxn_smiles.split(">>")
        
        if len(rxn_parts) != 2:
            return False
            
        reactants_smiles = rxn_parts[0]
        products_smiles = rxn_parts[1]
        
        try:
            # Parse reactants and products
            reactant_mols = []
            for smi in reactants_smiles.split("."):
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    reactant_mols.append(mol)
            
            product_mols = []
            for smi in products_smiles.split("."):
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    product_mols.append(mol)
            
            if not reactant_mols or not product_mols:
                return False
            
            # Get the protecting group pattern
            if self.protecting_group not in self.pg_patterns:
                return False
                
            pg_pattern = Chem.MolFromSmarts(self.pg_patterns[self.protecting_group])
            if pg_pattern is None:
                return False
            
            # Count protecting group occurrences in reactants and products
            reactant_pg_count = sum(len(mol.GetSubstructMatches(pg_pattern)) for mol in reactant_mols)
            product_pg_count = sum(len(mol.GetSubstructMatches(pg_pattern)) for mol in product_mols)
            
            # Check if the operation matches what we're looking for
            if self.operation == "deprotection":
                # Deprotection: protecting group count decreases
                return reactant_pg_count > product_pg_count
            elif self.operation == "protection":
                # Protection: protecting group count increases
                return product_pg_count > reactant_pg_count
            
        except Exception:
            return False
        
        return False
