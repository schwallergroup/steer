"""Generated evaluation code for: Early azide reduction to amine"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlyAzideReduction(BaseScoring):
    """
    Evaluates synthesis routes for early azide reduction to amine reactions.
    
    Checks if an azide functional group ([N-]=[N+]=[N-]) is reduced to a primary amine
    early in the synthesis sequence. Returns higher scores for earlier reductions.
    """
    
    def __init__(self, config: Dict):
        self.substrate_pattern = config.get("substrate_pattern", "[N-]=[N+]=[N-]")
        self.azide_mol = Chem.MolFromSmarts(self.substrate_pattern)
        self.primary_amine_pattern = Chem.MolFromSmarts("[NH2]")
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Azide reduction doesn't happen
        else:
            return 1 - x  # Earlier reduction gets higher score
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction node represents azide reduction to amine.
        
        Returns True if:
        1. Reactants contain azide functional group
        2. Products contain primary amine 
        3. Azide is consumed (not present in products)
        """
        rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
        if not rxn_smiles or ">>" not in rxn_smiles:
            return False
            
        try:
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse reactants
            reactant_mols = []
            for r_smiles in reactants_smiles.split("."):
                mol = Chem.MolFromSmiles(r_smiles.strip())
                if mol is not None:
                    reactant_mols.append(mol)
            
            # Parse products  
            product_mols = []
            for p_smiles in products_smiles.split("."):
                mol = Chem.MolFromSmiles(p_smiles.strip())
                if mol is not None:
                    product_mols.append(mol)
            
            if not reactant_mols or not product_mols:
                return False
            
            # Check if any reactant has azide group
            has_azide_reactant = any(
                mol.HasSubstructMatch(self.azide_mol) for mol in reactant_mols
            )
            
            if not has_azide_reactant:
                return False
            
            # Check if any product has primary amine
            has_amine_product = any(
                mol.HasSubstructMatch(self.primary_amine_pattern) for mol in product_mols
            )
            
            if not has_amine_product:
                return False
            
            # Check that azide is consumed (not present in products)
            has_azide_product = any(
                mol.HasSubstructMatch(self.azide_mol) for mol in product_mols
            )
            
            # Return True if azide was consumed and amine was formed
            return has_azide_reactant and has_amine_product and not has_azide_product
            
        except Exception:
            return False
