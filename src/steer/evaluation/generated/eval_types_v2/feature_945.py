"""Generated evaluation code for: Late stage thioether oxidation to sulfone"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageThioetherOxidation(BaseScoring):
    """
    Evaluates whether thioether oxidation to sulfone occurs at a late stage in the synthesis.
    Returns higher scores when the oxidation happens closer to the final product.
    """
    
    def __init__(self, config: Dict):
        # Define SMARTS patterns for thioether and sulfone
        self.thioether_pattern = Chem.MolFromSmarts("[#6]-S-[#6]")  # C-S-C
        self.sulfone_pattern = Chem.MolFromSmarts("[#6]-S(=O)(=O)-[#6]")  # C-SO2-C
    
    def route_scoring(self, x) -> float:
        """
        Convert depth fraction to score (0-10).
        Lower depth fractions (later stage) get higher scores.
        """
        if x < 0:
            return 0  # Oxidation doesn't happen
        else:
            # Late-stage oxidation is better, so invert the depth fraction
            return 10 * (1 - x)
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction represents thioether oxidation to sulfone.
        """
        try:
            rxn = d["metadata"]["mapped_reaction_smiles"].split(">>")
            reactants_smiles = rxn[0]
            products_smiles = rxn[1]
            
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            
            # Remove None molecules (parsing failures)
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            # Check if any reactant has thioether and any product has sulfone
            has_thioether_reactant = any(mol.HasSubstructMatch(self.thioether_pattern) for mol in reactants)
            has_sulfone_product = any(mol.HasSubstructMatch(self.sulfone_pattern) for mol in products)
            
            # Additional check: ensure we're not just detecting pre-existing sulfones
            has_sulfone_reactant = any(mol.HasSubstructMatch(self.sulfone_pattern) for mol in reactants)
            
            # This is thioether oxidation if:
            # 1. There's a thioether in reactants
            # 2. There's a sulfone in products  
            # 3. There wasn't already a sulfone in reactants (new sulfone formation)
            return has_thioether_reactant and has_sulfone_product and not has_sulfone_reactant
            
        except (KeyError, IndexError, AttributeError):
            return False
