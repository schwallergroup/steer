"""Generated evaluation code for: Late stage Weinreb amide to aldehyde reduction"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class WeinrebAmideReduction(BaseScoring):
    """
    Evaluates whether a Weinreb amide to aldehyde reduction occurs at late stage.
    
    Checks for the transformation of Weinreb amide [C(=O)N(CH3)OH] to aldehyde [C(=O)H]
    and penalizes if it doesn't occur late enough in the synthesis (after stage_threshold).
    """
    
    def __init__(self, config: Dict):
        self.reaction_smarts = config["parameters"]["reaction_smarts"]
        self.timing = config["parameters"]["timing"]
        self.stage_threshold = config["parameters"]["stage_threshold"]
        
        # Parse the reaction SMARTS to get reactant and product patterns
        reactant_smarts, product_smarts = self.reaction_smarts.split(">>")
        self.reactant_pattern = Chem.MolFromSmarts(reactant_smarts)
        self.product_pattern = Chem.MolFromSmarts(product_smarts)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Reaction doesn't occur in route
        
        if self.timing == "late":
            # For late-stage reactions, penalize early occurrence
            if x >= self.stage_threshold:
                return 1.0  # Good - occurs late in synthesis
            else:
                # Linear penalty for occurring too early
                return x / self.stage_threshold
        else:
            # For early-stage reactions (if needed), reverse the logic
            if x <= (1 - self.stage_threshold):
                return 1.0
            else:
                return (1 - x) / self.stage_threshold
    
    def hit_condition(self, d) -> bool:
        """
        Check if the current reaction node represents a Weinreb amide reduction.
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            # Split reaction into reactants and products
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
            
            # Check if any reactant contains Weinreb amide pattern
            has_weinreb_reactant = any(
                mol.HasSubstructMatch(self.reactant_pattern) 
                for mol in reactant_mols
            )
            
            # Check if any product contains aldehyde pattern
            has_aldehyde_product = any(
                mol.HasSubstructMatch(self.product_pattern) 
                for mol in product_mols
            )
            
            return has_weinreb_reactant and has_aldehyde_product
            
        except Exception:
            return False
