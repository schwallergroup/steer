"""Generated evaluation code for: Methylenedioxy protection strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class MethylenedioxyProtection(BaseScoring):
    """
    Evaluates synthesis routes based on the use of methylenedioxy protection strategy.
    
    This class checks for the formation of methylenedioxy-protected catechols using
    dibromomethane as the cyclization reagent. The protection involves converting
    catechol (1,2-dihydroxybenzene) to a methylenedioxy bridge structure.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "float")
        self.target_depth = config.get("target_depth", {}).get("value", 0.5)
    
    def route_scoring(self, x) -> float:
        """
        Convert depth fraction to score (0-10 scale).
        Earlier protection (lower depth) is generally preferred.
        """
        if x < 0:
            return 0  # Protection strategy not found
        
        if self.condition_type == "bool":
            return 1  # Protection strategy found
        else:
            # Earlier protection is better (inverse relationship with depth)
            return max(0, 1 - x)
    
    def hit_condition(self, d) -> bool:
        """
        Check if a single reaction node represents methylenedioxy protection formation.
        
        Looks for:
        1. Catechol substructure in reactants
        2. Methylenedioxy bridge in products
        3. Dibromomethane as reagent
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
        
        try:
            reactants_smiles, products_smiles = mapped_rxn.split(">>")
            reactant_mols = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            product_mols = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            # Filter out None molecules
            reactant_mols = [mol for mol in reactant_mols if mol is not None]
            product_mols = [mol for mol in product_mols if mol is not None]
            
            if not reactant_mols or not product_mols:
                return False
            
            # Check for dibromomethane reagent
            dibromomethane_pattern = Chem.MolFromSmarts("BrCBr")
            has_dibromomethane = any(mol.HasSubstructMatch(dibromomethane_pattern) for mol in reactant_mols)
            
            # Check for catechol pattern in reactants (ortho-dihydroxybenzene)
            catechol_pattern = Chem.MolFromSmarts("c1ccccc1[OH][OH]")  # Two OH groups on benzene
            ortho_diol_pattern = Chem.MolFromSmarts("c1c([OH])c([OH])ccc1")  # More specific ortho pattern
            
            has_catechol = any(
                mol.HasSubstructMatch(catechol_pattern) or mol.HasSubstructMatch(ortho_diol_pattern)
                for mol in reactant_mols
            )
            
            # Check for methylenedioxy bridge in products
            methylenedioxy_pattern = Chem.MolFromSmarts("c1c2c(ccc1)OCO2")  # Methylenedioxy bridge
            benzodioxole_pattern = Chem.MolFromSmarts("O1COc2ccccc21")  # Alternative pattern
            
            has_methylenedioxy = any(
                mol.HasSubstructMatch(methylenedioxy_pattern) or mol.HasSubstructMatch(benzodioxole_pattern)
                for mol in product_mols
            )
            
            # All conditions must be met
            return has_dibromomethane and has_catechol and has_methylenedioxy
            
        except Exception:
            return False
