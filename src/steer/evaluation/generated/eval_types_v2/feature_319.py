"""Generated evaluation code for: Early Sandmeyer reaction for aryl bromide formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlySandmeyerReaction(BaseScoring):
    """
    Evaluates whether a Sandmeyer reaction (aniline to aryl bromide conversion) 
    occurs early in the synthesis route.
    """
    
    def __init__(self, config: Dict):
        self.target_stage = config.get("stage", "early")
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Sandmeyer reaction doesn't occur
        else:
            # Early stage is better - lower depth fraction gives higher score
            return 1 - x
    
    def hit_condition(self, d) -> bool:
        """
        Checks if the reaction node represents a Sandmeyer reaction
        (aniline derivative to aryl bromide conversion).
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
                
            reactants, products = rxn_smiles.split(">>")
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".")]
            
            # Filter out None molecules
            reactant_mols = [mol for mol in reactant_mols if mol is not None]
            product_mols = [mol for mol in product_mols if mol is not None]
            
            # Check for aniline pattern in reactants (aromatic amine)
            aniline_pattern = Chem.MolFromSmarts("[c:1][NH2:2]")
            has_aniline = any(mol.HasSubstructMatch(aniline_pattern) for mol in reactant_mols)
            
            # Check for aryl bromide pattern in products
            aryl_bromide_pattern = Chem.MolFromSmarts("[c:1][Br:2]")
            has_aryl_bromide = any(mol.HasSubstructMatch(aryl_bromide_pattern) for mol in product_mols)
            
            # Additional check for typical Sandmeyer conditions (presence of CuBr or similar)
            # Look for copper or bromide sources in reactants
            copper_pattern = Chem.MolFromSmarts("[Cu]")
            bromide_source = any("Br" in Chem.MolToSmiles(mol) for mol in reactant_mols if mol is not None)
            has_sandmeyer_reagents = any(mol.HasSubstructMatch(copper_pattern) for mol in reactant_mols) or bromide_source
            
            return has_aniline and has_aryl_bromide and has_sandmeyer_reagents
            
        except Exception:
            return False
