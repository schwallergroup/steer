"""Generated evaluation code for: Ester protecting group for carboxylic acid"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EsterProtectingGroup(BaseScoring):
    """
    Evaluates synthesis routes for the use of ethyl ester protecting groups on carboxylic acids.
    Checks if carboxylic acid is protected as ethyl ester at the specified stage (early/late).
    """
    
    def __init__(self, config: Dict):
        self.protected_group_smarts = config["parameters"]["protected_group"]  # "C(=O)O"
        self.protecting_group_smarts = config["parameters"]["protecting_group"]  # "CCOC(=O)" 
        self.target_stage = config["parameters"]["stage"]  # "early" or "late"
        
        # Convert to RDKit patterns
        self.protected_pattern = Chem.MolFromSmarts(self.protected_group_smarts)
        self.protecting_pattern = Chem.MolFromSmarts(self.protecting_group_smarts)

    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Protection strategy not used
        
        # Convert depth fraction to preference based on target stage
        if self.target_stage == "early":
            # Early protection is preferred (lower depth fraction is better)
            return 1 - x
        else:  # late stage
            # Late protection is preferred (higher depth fraction is better) 
            return x

    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves protecting a carboxylic acid with ethyl ester.
        """
        rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
        if not rxn_smiles or ">>" not in rxn_smiles:
            return False
            
        reactants_smiles, products_smiles = rxn_smiles.split(">>")
        
        # Parse reactants and products
        reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
        products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
        
        reactants = [mol for mol in reactants if mol is not None]
        products = [mol for mol in products if mol is not None]
        
        if not reactants or not products:
            return False
        
        # Check if any reactant has carboxylic acid and any product has ethyl ester
        has_carboxylic_acid_reactant = any(
            mol.HasSubstructMatch(self.protected_pattern) for mol in reactants
        )
        
        has_ethyl_ester_product = any(
            mol.HasSubstructMatch(self.protecting_pattern) for mol in products
        )
        
        # Also check if ethyl ester protecting reagent is present in reactants
        ethyl_reagents = ["CCO", "CCOH", "CC(=O)OCC"]  # Common ethylating agents
        has_ethyl_reagent = any(
            any(Chem.MolFromSmiles(reagent_smi).HasSubstructMatch(mol) 
                for reagent_smi in ethyl_reagents if Chem.MolFromSmiles(reagent_smi))
            for mol in reactants
        )
        
        return (has_carboxylic_acid_reactant and 
                has_ethyl_ester_product and 
                has_ethyl_reagent)
