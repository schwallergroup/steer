"""Generated evaluation code for: TBDPS protecting group for primary alcohol"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class TBDPSProtectingGroup(BaseScoring):
    """
    Evaluates synthesis routes for TBDPS protection of primary alcohols at mid-stage.
    Checks for the presence of TBDPS protection reaction and penalizes if it occurs
    too early or too late in the synthesis.
    """
    
    def __init__(self, config: Dict):
        self.protecting_group = config["parameters"]["protecting_group"]
        self.functional_group = config["parameters"]["functional_group"]
        self.timing = config["parameters"]["timing"]
        
        # Define target depth range for mid-stage (0.3-0.7 of total depth)
        self.optimal_min = 0.3
        self.optimal_max = 0.7
        
    def route_scoring(self, x) -> float:
        """
        Score based on timing of TBDPS protection.
        Returns 0-10 where 10 is optimal mid-stage timing.
        """
        if x < 0:
            return 0  # Protection not found
            
        if self.optimal_min <= x <= self.optimal_max:
            return 10  # Optimal mid-stage timing
        elif x < self.optimal_min:
            # Too late in synthesis
            return 5 * (x / self.optimal_min)
        else:
            # Too early in synthesis  
            return 5 * (1 - (x - self.optimal_max) / (1 - self.optimal_max))
    
    def hit_condition(self, d) -> bool:
        """
        Check if reaction involves TBDPS protection of primary alcohol.
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles:
                return False
                
            parts = rxn_smiles.split(">>")
            if len(parts) != 2:
                return False
                
            reactants = parts[0]
            products = parts[1]
            
            # Check for TBDPS introduction (Si with tert-butyl and diphenyl groups)
            tbdps_pattern = "[Si]([CH3]([CH3])[CH3])(c1ccccc1)(c2ccccc2)"
            primary_alcohol_pattern = "[CH2][OH]"
            protected_pattern = "[CH2]O[Si]"
            
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".")]
            
            # Filter out None molecules
            reactant_mols = [mol for mol in reactant_mols if mol is not None]
            product_mols = [mol for mol in product_mols if mol is not None]
            
            # Check if reactants contain primary alcohol
            has_primary_alcohol_reactant = any(
                mol.HasSubstructMatch(Chem.MolFromSmarts(primary_alcohol_pattern))
                for mol in reactant_mols
            )
            
            # Check if products contain TBDPS-protected alcohol
            has_tbdps_product = any(
                mol.HasSubstructMatch(Chem.MolFromSmarts(tbdps_pattern)) and
                mol.HasSubstructMatch(Chem.MolFromSmarts(protected_pattern))
                for mol in product_mols
            )
            
            # Check if TBDPS reagent is present in reactants
            has_tbdps_reagent = any(
                mol.HasSubstructMatch(Chem.MolFromSmarts(tbdps_pattern))
                for mol in reactant_mols
            )
            
            return (has_primary_alcohol_reactant and 
                   has_tbdps_product and 
                   has_tbdps_reagent)
                   
        except Exception:
            return False
