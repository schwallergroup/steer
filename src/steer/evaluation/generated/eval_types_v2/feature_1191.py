"""Generated evaluation code for: Azide intermediate for amine installation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class AzideIntermediateForAmine(BaseScoring):
    """
    Checks if the synthesis route uses azide as an intermediate that is later reduced 
    to install amine functionality. Detects the presence of azide groups and their 
    subsequent reduction to amines.
    """
    
    def __init__(self, config: Dict):
        self.intermediate_smarts = config["parameters"]["intermediate_smarts"]
        self.azide_pattern = Chem.MolFromSmarts(self.intermediate_smarts)
        # Pattern for primary amine that could result from azide reduction
        self.amine_pattern = Chem.MolFromSmarts("[C]N")
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Condition not met
        else:
            return 1 - x  # Earlier use of azide intermediate is better
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves azide reduction to amine installation.
        Looks for azide group in reactants and corresponding amine in products.
        """
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            reactants = [Chem.MolFromSmiles(r) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p) for p in products_smiles.split(".")]
            
            # Check if any reactant contains azide group
            has_azide_reactant = any(
                mol and mol.HasSubstructMatch(self.azide_pattern) 
                for mol in reactants
            )
            
            # Check if any product contains amine group
            has_amine_product = any(
                mol and mol.HasSubstructMatch(self.amine_pattern) 
                for mol in products
            )
            
            # Check if we're converting azide to amine (azide in reactant, amine in product)
            if has_azide_reactant and has_amine_product:
                # Additional check: ensure azide is being consumed
                has_azide_product = any(
                    mol and mol.HasSubstructMatch(self.azide_pattern) 
                    for mol in products
                )
                # Return True if azide is consumed (not present in products)
                return not has_azide_product
            
            return False
            
        except Exception:
            return False
