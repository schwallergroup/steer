"""Generated evaluation code for: Early azide-amine conversion strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlyAzideAmineConversion(BaseScoring):
    """
    Evaluates whether azide-to-amine reduction occurs early in the synthesis route.
    Rewards routes where azide reduction happens within the specified depth threshold.
    """
    
    def __init__(self, config: Dict):
        self.depth_threshold = config["parameters"].get("depth_threshold", 2)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Azide reduction doesn't occur
        
        # Convert depth fraction to score (0-10)
        # Early occurrence (low depth fraction) gets higher score
        if x <= self.depth_threshold / 10.0:  # Within early threshold
            return 10
        else:
            # Linearly decrease score for later occurrences
            return max(0, 10 - (x * 10))
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction involves azide reduction to amine."""
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            rxn_parts = rxn_smiles.split(">>")
            
            if len(rxn_parts) != 2:
                return False
                
            products = rxn_parts[0]
            reactants = rxn_parts[1]
            
            # Parse molecules
            prod_mol = Chem.MolFromSmiles(products)
            react_mols = [Chem.MolFromSmiles(r) for r in reactants.split(".")]
            
            if not prod_mol or not all(react_mols):
                return False
            
            # Define azide and amine patterns
            azide_pattern = Chem.MolFromSmarts("[N-]=[N+]=[N-]")  # Azide functional group
            primary_amine_pattern = Chem.MolFromSmarts("[CH2,CH][NH2]")  # Primary amine
            
            # Check if reactants contain azide
            has_azide_reactant = any(mol.HasSubstructMatch(azide_pattern) for mol in react_mols)
            
            # Check if product contains primary amine
            has_amine_product = prod_mol.HasSubstructMatch(primary_amine_pattern)
            
            # Additional check: ensure azide is actually being reduced
            # Count azide groups in reactants vs products
            reactant_azides = sum(len(mol.GetSubstructMatches(azide_pattern)) for mol in react_mols)
            product_azides = len(prod_mol.GetSubstructMatches(azide_pattern))
            
            # True if we have azide in reactants, amine in product, and fewer azides in product
            return (has_azide_reactant and has_amine_product and 
                   reactant_azides > product_azides)
            
        except Exception:
            return False
