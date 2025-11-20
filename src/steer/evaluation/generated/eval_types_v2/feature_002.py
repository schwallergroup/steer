"""Generated evaluation code for: Late stage amide reduction to methylene"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageAmideReduction(BaseScoring):
    """
    Evaluates whether an amide reduction to methylene occurs as the final step.
    Checks for the transformation of C(=O)N to CN in the last reaction.
    """
    
    def __init__(self, config: Dict):
        self.substrate_pattern = config["parameters"]["substrate_carbonyl"]  # "C(=O)N"
        self.product_pattern = config["parameters"]["product_methylene"]     # "CN"
        self.timing = config["parameters"]["timing"]  # "final_step"
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Reaction doesn't happen
        else:
            if self.timing == "final_step":
                # For final step, we want x to be close to 1.0 (very late)
                return 10 * x  # Score 0-10, higher for later occurrence
            else:
                # For other timing preferences, could implement different scoring
                return 10 * (1 - x)
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction represents amide reduction to methylene.
        """
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            
            # Remove None molecules (parsing failures)
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            if not reactants or not products:
                return False
            
            # Create SMARTS patterns for substructure matching
            amide_pattern = Chem.MolFromSmarts(self.substrate_pattern)
            methylene_pattern = Chem.MolFromSmarts(self.product_pattern)
            
            if amide_pattern is None or methylene_pattern is None:
                return False
            
            # Check if any reactant contains the amide pattern
            has_amide_reactant = any(mol.HasSubstructMatch(amide_pattern) for mol in reactants)
            
            # Check if any product contains the methylene pattern where amide was reduced
            has_methylene_product = any(mol.HasSubstructMatch(methylene_pattern) for mol in products)
            
            # Additional check: ensure we're not just detecting coincidental patterns
            # The main product should have the methylene but not the amide
            main_product = products[0]  # Assume first product is main product
            has_reduced_amide = (main_product.HasSubstructMatch(methylene_pattern) and 
                               not main_product.HasSubstructMatch(amide_pattern))
            
            return has_amide_reactant and has_reduced_amide
            
        except Exception:
            return False
