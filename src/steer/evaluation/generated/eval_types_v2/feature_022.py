"""Generated evaluation code for: Late stage Gabriel amine deprotection"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class GabrielAmineDeprotection(BaseScoring):
    """
    Evaluates synthesis routes for late-stage Gabriel amine deprotection.
    Checks if phthalimide-protected amine is deprotected to reveal primary amine
    at a late stage in the synthesis.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "float")
        self.target_depth = config.get("target_depth", {}).get("value", 0.1)  # Default to very late stage
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Deprotection doesn't happen
        else:
            if self.condition_type == "bool":
                return 1 if x <= self.target_depth else 0
            else:
                # Late-stage deprotection is better (lower depth fraction)
                return max(0, 10 * (1 - x))
    
    def hit_condition(self, d):
        """
        Check if this reaction involves Gabriel amine deprotection
        (phthalimide -> primary amine conversion)
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
                
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse reactants and products
            reactant_mols = []
            for smiles in reactants_smiles.split("."):
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    reactant_mols.append(mol)
                    
            product_mols = []
            for smiles in products_smiles.split("."):
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    product_mols.append(mol)
            
            if not reactant_mols or not product_mols:
                return False
            
            # Define phthalimide pattern (N-substituted phthalimide)
            phthalimide_pattern = Chem.MolFromSmarts("[#6]1[#6][#6][#6]2[#6]([#6]1)[#6](=[#8])[#7]([*])[#6]2=[#8]")
            
            # Define primary amine pattern
            primary_amine_pattern = Chem.MolFromSmarts("[#6][#7;H2]")
            
            # Check if any reactant contains phthalimide
            has_phthalimide_reactant = any(
                mol.HasSubstructMatch(phthalimide_pattern) for mol in reactant_mols
            )
            
            # Check if any product contains primary amine
            has_primary_amine_product = any(
                mol.HasSubstructMatch(primary_amine_pattern) for mol in product_mols
            )
            
            # Additional check: look for phthalic acid or phthalimide byproducts
            phthalic_acid_pattern = Chem.MolFromSmarts("c1ccc2c(c1)C(=O)O")  # Simplified phthalic acid derivative
            phthalimide_byproduct = Chem.MolFromSmarts("c1ccc2c(c1)C(=O)NC2=O")  # Phthalimide
            
            has_phthalic_byproduct = any(
                mol.HasSubstructMatch(phthalic_acid_pattern) or mol.HasSubstructMatch(phthalimide_byproduct)
                for mol in product_mols
            )
            
            return has_phthalimide_reactant and has_primary_amine_product and has_phthalic_byproduct
            
        except Exception:
            return False
