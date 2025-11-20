"""Generated evaluation code for: Early stage Fischer indole synthesis"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlyFischerIndole(BaseScoring):
    """
    Evaluates routes for early-stage Fischer indole synthesis.
    
    The Fischer indole synthesis involves the reaction of a phenylhydrazine 
    with a ketone or aldehyde to form an indole ring system. This class
    checks for the presence of this reaction pattern and rewards routes
    where it occurs early in the synthesis.
    """
    
    def __init__(self, config: Dict):
        self.timing_weight = config.get("timing_weight", 1.0)
        # Fischer indole involves phenylhydrazine + carbonyl -> indole
        self.phenylhydrazine_pattern = Chem.MolFromSmarts("[NH2]-[NH]-c1ccccc1")
        self.indole_pattern = Chem.MolFromSmarts("c1ccc2[nH]ccc2c1")
        self.carbonyl_pattern = Chem.MolFromSmarts("[C]=[O]")
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Fischer indole synthesis doesn't occur
        else:
            # Early stage is better - invert the depth fraction
            return (1 - x) * 10 * self.timing_weight
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction represents a Fischer indole synthesis.
        
        Criteria:
        1. Reactants contain phenylhydrazine and carbonyl patterns
        2. Product contains indole pattern
        3. Indole pattern is newly formed (not present in reactants)
        """
        rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
        if not rxn_smiles or ">>" not in rxn_smiles:
            return False
            
        try:
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse reactants and products
            reactant_mols = []
            for r_smiles in reactants_smiles.split("."):
                mol = Chem.MolFromSmiles(r_smiles)
                if mol is not None:
                    reactant_mols.append(mol)
            
            product_mols = []
            for p_smiles in products_smiles.split("."):
                mol = Chem.MolFromSmiles(p_smiles)
                if mol is not None:
                    product_mols.append(mol)
            
            if not reactant_mols or not product_mols:
                return False
            
            # Check if reactants contain Fischer indole precursors
            has_phenylhydrazine = any(
                mol.HasSubstructMatch(self.phenylhydrazine_pattern) 
                for mol in reactant_mols
            )
            
            has_carbonyl = any(
                mol.HasSubstructMatch(self.carbonyl_pattern)
                for mol in reactant_mols
            )
            
            # Check if product contains indole
            has_indole_product = any(
                mol.HasSubstructMatch(self.indole_pattern)
                for mol in product_mols
            )
            
            # Check that indole is newly formed (not in reactants)
            indole_in_reactants = any(
                mol.HasSubstructMatch(self.indole_pattern)
                for mol in reactant_mols
            )
            
            # Fischer indole synthesis detected if:
            # - Has phenylhydrazine and carbonyl in reactants
            # - Forms indole in product
            # - Indole was not present in reactants
            return (has_phenylhydrazine and has_carbonyl and 
                   has_indole_product and not indole_in_reactants)
            
        except Exception:
            return False
