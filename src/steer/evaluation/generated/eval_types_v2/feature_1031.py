"""Generated evaluation code for: Late stage THP phenol deprotection"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class THPPhenolDeprotection(BaseScoring):
    """
    Evaluates late-stage THP (tetrahydropyranyl) phenol deprotection in synthesis routes.
    Returns higher scores when THP deprotection occurs closer to the final step.
    """
    
    def __init__(self, config: Dict):
        self.timing = config.get("timing", "late")
        self.operation = config.get("operation", "deprotection")
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # THP deprotection doesn't happen
        else:
            # Late-stage deprotection is better (closer to 0 depth)
            # Scale from 0-10 where 10 is latest possible
            return (1 - x) * 10
    
    def hit_condition(self, d):
        """
        Check if this reaction performs THP phenol deprotection.
        Looks for THP-protected phenol in reactants and free phenol in products.
        """
        rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
        if not rxn_smiles or ">>" not in rxn_smiles:
            return False
        
        try:
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".") if r.strip()]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".") if p.strip()]
            
            # Filter out None molecules
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            if not reactants or not products:
                return False
            
            # THP-protected phenol pattern: phenol oxygen connected to THP ring
            thp_phenol_pattern = Chem.MolFromSmarts("[OH0]([c])C1CCCCO1")
            
            # Free phenol pattern
            phenol_pattern = Chem.MolFromSmarts("[OH1][c]")
            
            # Check if any reactant has THP-protected phenol
            has_thp_phenol_reactant = any(mol.HasSubstructMatch(thp_phenol_pattern) for mol in reactants)
            
            # Check if any product has free phenol
            has_phenol_product = any(mol.HasSubstructMatch(phenol_pattern) for mol in products)
            
            # Additional check: THP leaving group in products (tetrahydropyran-2-ol or related)
            thp_leaving_group = Chem.MolFromSmarts("C1CCCCO1")
            has_thp_leaving = any(mol.HasSubstructMatch(thp_leaving_group) for mol in products)
            
            return has_thp_phenol_reactant and has_phenol_product and has_thp_leaving
            
        except Exception:
            return False
