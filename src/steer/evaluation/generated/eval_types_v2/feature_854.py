"""Generated evaluation code for: Ester protecting group strategy for carboxylic acid"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EsterProtectingGroupStrategy(BaseScoring):
    """
    Evaluates whether an ethyl ester protecting group strategy is used for carboxylic acids.
    Checks if ethyl ester hydrolysis (deprotection) occurs at a late stage in the synthesis.
    """
    
    def __init__(self, config: Dict):
        self.deprotection_stage = config["parameters"]["deprotection_stage"]
        # SMARTS pattern for ethyl ester hydrolysis (ethyl ester -> carboxylic acid)
        self.ethyl_ester_pattern = "[C:1](=[O:2])[O:3][CH2:4][CH3:5]"
        self.carboxylic_acid_pattern = "[C:1](=[O:2])[OH:3]"
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # No ethyl ester deprotection found
        
        if self.deprotection_stage == "late":
            # Late-stage deprotection is better (higher depth fraction)
            return x * 10  # Convert to 0-10 scale, rewarding later deprotection
        elif self.deprotection_stage == "early":
            # Early-stage deprotection is better (lower depth fraction)
            return (1 - x) * 10
        else:
            # Any deprotection is good
            return 10 if x >= 0 else 0
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves ethyl ester hydrolysis to carboxylic acid.
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
                
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse reactants and products
            reactant_mols = []
            for r_smiles in reactants_smiles.split("."):
                mol = Chem.MolFromSmiles(r_smiles)
                if mol:
                    reactant_mols.append(mol)
            
            product_mols = []
            for p_smiles in products_smiles.split("."):
                mol = Chem.MolFromSmiles(p_smiles)
                if mol:
                    product_mols.append(mol)
            
            if not reactant_mols or not product_mols:
                return False
            
            # Check if any reactant has ethyl ester and any product has carboxylic acid
            ethyl_ester_in_reactants = any(
                mol.HasSubstructMatch(Chem.MolFromSmarts(self.ethyl_ester_pattern))
                for mol in reactant_mols
            )
            
            carboxylic_acid_in_products = any(
                mol.HasSubstructMatch(Chem.MolFromSmarts(self.carboxylic_acid_pattern))
                for mol in product_mols
            )
            
            # Additional check: ethyl ester should not be present in products
            # (confirming it was hydrolyzed)
            ethyl_ester_in_products = any(
                mol.HasSubstructMatch(Chem.MolFromSmarts(self.ethyl_ester_pattern))
                for mol in product_mols
            )
            
            return (ethyl_ester_in_reactants and 
                   carboxylic_acid_in_products and 
                   not ethyl_ester_in_products)
                   
        except Exception:
            return False
