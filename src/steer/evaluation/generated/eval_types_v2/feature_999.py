"""Generated evaluation code for: Final step alkyne reduction to alkane"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class FinalStepAlkyneReduction(BaseScoring):
    """
    Checks if the final step (depth 1) is an alkyne reduction to alkane via hydrogenation.
    This evaluates whether the synthetic route strategically saves the alkyne reduction 
    for the final step to avoid carrying saturated chains through earlier transformations.
    """
    
    def __init__(self, config: Dict):
        self.step = config.get("step", 1)  # Target step depth
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Condition not met
        elif x == 1.0:  # Final step (depth 1)
            return 1.0  # Perfect score
        else:
            return 0  # Not in final step
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction is an alkyne to alkane hydrogenation"""
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse molecules
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            
            if not all(products) or not all(reactants):
                return False
            
            # Check for alkyne in reactants and alkane in products
            alkyne_pattern = Chem.MolFromSmarts("C#C")  # Terminal or internal alkyne
            alkane_pattern = Chem.MolFromSmarts("CC")   # Alkane pattern
            
            # Find alkyne in reactants
            has_alkyne_reactant = any(mol.HasSubstructMatch(alkyne_pattern) for mol in reactants)
            
            # Check if products have corresponding alkane and no alkyne
            has_alkane_product = any(mol.HasSubstructMatch(alkane_pattern) for mol in products)
            has_alkyne_product = any(mol.HasSubstructMatch(alkyne_pattern) for mol in products)
            
            # Verify this is a reduction (alkyne -> alkane, no alkyne remaining)
            if has_alkyne_reactant and has_alkane_product and not has_alkyne_product:
                # Additional check: look for hydrogen addition pattern
                # Count heavy atoms to ensure we're not breaking/forming C-C bonds
                reactant_heavy_atoms = sum(mol.GetNumHeavyAtoms() for mol in reactants if mol.GetNumHeavyAtoms() > 2)
                product_heavy_atoms = sum(mol.GetNumHeavyAtoms() for mol in products if mol.GetNumHeavyAtoms() > 2)
                
                return reactant_heavy_atoms == product_heavy_atoms
                
            return False
            
        except Exception:
            return False
