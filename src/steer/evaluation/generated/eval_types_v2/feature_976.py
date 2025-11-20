"""Generated evaluation code for: Benzyl ether protecting group strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BenzylEtherProtection(BaseScoring):
    """
    Evaluates synthesis routes based on benzyl ether protecting group strategy for phenols.
    Checks if benzyl ether protection occurs early in the synthesis and deprotection 
    (typically via hydrogenation) occurs at an appropriate later stage.
    """
    
    def __init__(self, config: Dict):
        self.strategy = config.get("protecting_group", "benzyl_ether")
        self.functional_group = config.get("functional_group", "phenol")
        self.deprotection_method = config.get("deprotection_method", "hydrogenation")
        
        # SMARTS patterns for detection
        self.benzyl_ether_pattern = "[OH1][c]"  # Phenol pattern
        self.protected_pattern = "[O][CH2][c]1[cH][cH][cH][cH][cH]1"  # Benzyl ether pattern
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Protection strategy not used
        else:
            # Earlier protection is better (closer to 1.0 gets higher score)
            # Score ranges from 1-10, with early protection getting higher scores
            return 1 + (9 * (1 - x))
    
    def hit_condition(self, d):
        """
        Check if this reaction involves benzyl ether protection of a phenol.
        Returns True if a phenol is converted to a benzyl ether.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            reactants_smiles, products_smiles = mapped_rxn.split(">>")
            
            # Parse reactants and products
            reactant_mols = []
            for r_smiles in reactants_smiles.split("."):
                mol = Chem.MolFromSmiles(r_smiles.strip())
                if mol is not None:
                    reactant_mols.append(mol)
            
            product_mols = []
            for p_smiles in products_smiles.split("."):
                mol = Chem.MolFromSmiles(p_smiles.strip())
                if mol is not None:
                    product_mols.append(mol)
            
            # Check for phenol in reactants and benzyl ether in products
            phenol_in_reactants = any(
                mol.HasSubstructMatch(Chem.MolFromSmarts(self.benzyl_ether_pattern)) 
                for mol in reactant_mols
            )
            
            benzyl_ether_in_products = any(
                mol.HasSubstructMatch(Chem.MolFromSmarts(self.protected_pattern))
                for mol in product_mols
            )
            
            # Also check for benzyl bromide/chloride reagent presence
            benzyl_reagent_pattern = "[CH2]([Cl,Br])[c]1[cH][cH][cH][cH][cH]1"
            benzyl_reagent_present = any(
                mol.HasSubstructMatch(Chem.MolFromSmarts(benzyl_reagent_pattern))
                for mol in reactant_mols
            )
            
            return (phenol_in_reactants and 
                   benzyl_ether_in_products and 
                   benzyl_reagent_present)
            
        except Exception:
            return False
