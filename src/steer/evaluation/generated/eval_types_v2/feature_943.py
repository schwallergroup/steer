"""Generated evaluation code for: Late stage sulfide oxidation to sulfone"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageSulfideOxidation(BaseScoring):
    """
    Evaluates whether sulfide oxidation to sulfone occurs at a late stage in the synthesis.
    Checks for conversion of sulfide [S] to sulfone S(=O)(=O) functionality.
    """
    
    def __init__(self, config: Dict):
        self.substrate_pattern = config.get("substrate_pattern", "[S]")
        self.product_pattern = config.get("product_pattern", "S(=O)(=O)")
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Sulfide oxidation doesn't happen
        else:
            return 1 - x  # Later stage oxidation is better (higher score for smaller x)
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction represents sulfide oxidation to sulfone"""
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
            
            # Create pattern molecules for substructure matching
            sulfide_pattern = Chem.MolFromSmarts(self.substrate_pattern)
            sulfone_pattern = Chem.MolFromSmarts(self.product_pattern)
            
            if sulfide_pattern is None or sulfone_pattern is None:
                return False
            
            # Check if any reactant contains sulfide pattern
            has_sulfide_reactant = any(mol.HasSubstructMatch(sulfide_pattern) for mol in reactants)
            
            # Check if any product contains sulfone pattern
            has_sulfone_product = any(mol.HasSubstructMatch(sulfone_pattern) for mol in products)
            
            # Must have sulfide in reactants and sulfone in products
            return has_sulfide_reactant and has_sulfone_product
            
        except (KeyError, ValueError, AttributeError):
            return False
