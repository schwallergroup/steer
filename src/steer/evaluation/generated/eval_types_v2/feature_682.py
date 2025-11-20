"""Generated evaluation code for: Multiple protecting group deprotections combined"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SimultaneousDeprotection(BaseScoring):
    """
    Checks for simultaneous deprotection of multiple protecting groups in a single reaction step.
    Scores based on whether the specified protecting groups are removed together.
    """
    
    def __init__(self, config: Dict):
        self.groups = config["parameters"]["groups"]
        self.strategy_type = config["parameters"]["strategy_type"]
        
        # Define SMARTS patterns for protecting groups
        self.group_patterns = {
            "Boc": "[NX3:1][CX3](=[OX1])[OX2][CX4]([CH3])([CH3])[CH3]",  # tert-butoxycarbonyl
            "tBu_ester": "[CX3](=[OX1])[OX2][CX4]([CH3])([CH3])[CH3]",    # tert-butyl ester
            "Cbz": "[NX3:1][CX3](=[OX1])[OX2][CH2][c1ccccc1]",           # benzyloxycarbonyl
            "TBDMS": "[OH1,NH1,NH2:1][Si]([CH3])([CH3])[CX4]([CH3])([CH3])[CH3]",  # tert-butyldimethylsilyl
            "Bn": "[NH1,NH2:1][CH2][c1ccccc1]",                          # benzyl
            "Ac": "[NX3:1][CX3](=[OX1])[CH3]"                            # acetyl
        }
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Strategy not found
        else:
            # Earlier simultaneous deprotection is better (more strategic)
            return (1 - x) * 10
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction simultaneously removes all specified protecting groups
        """
        try:
            rxn = d["metadata"]["mapped_reaction_smiles"].split(">>")
            reactants = rxn[0]
            products = rxn[1]
            
            reactant_mol = Chem.MolFromSmiles(reactants)
            product_mol = Chem.MolFromSmiles(products)
            
            if reactant_mol is None or product_mol is None:
                return False
            
            # Check if all specified protecting groups are present in reactants
            groups_in_reactants = []
            for group in self.groups:
                if group in self.group_patterns:
                    pattern = Chem.MolFromSmarts(self.group_patterns[group])
                    if pattern and reactant_mol.HasSubstructMatch(pattern):
                        groups_in_reactants.append(group)
            
            # Must have all specified groups in reactants
            if len(groups_in_reactants) != len(self.groups):
                return False
            
            # Check if all these groups are removed in products
            groups_removed = []
            for group in self.groups:
                if group in self.group_patterns:
                    pattern = Chem.MolFromSmarts(self.group_patterns[group])
                    if pattern:
                        # Group should be present in reactants but absent in products
                        if (reactant_mol.HasSubstructMatch(pattern) and 
                            not product_mol.HasSubstructMatch(pattern)):
                            groups_removed.append(group)
            
            # All specified groups must be simultaneously removed
            return len(groups_removed) == len(self.groups)
            
        except Exception:
            return False
