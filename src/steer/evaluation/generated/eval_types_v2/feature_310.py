"""Generated evaluation code for: Late stage Curtius rearrangement for aniline synthesis"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class CurtiusRearrangementDepth(BaseScoring):
    """
    Evaluates the depth at which a Curtius rearrangement occurs in a synthesis route.
    Curtius rearrangement converts carboxylic acids to amines via acyl azide intermediates,
    often used for late-stage aniline formation.
    """
    
    def __init__(self, config: Dict):
        self.timing = config.get("timing", "late")
        self.purpose = config.get("purpose", "aniline_formation")
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Curtius rearrangement not found
        
        if self.timing == "late":
            return 1 - x  # Late-stage preferred, so lower depth fraction is better
        else:
            return x  # Early-stage preferred
            
    def hit_condition(self, d) -> bool:
        """
        Detects Curtius rearrangement by looking for:
        1. Carboxylic acid -> primary amine conversion
        2. Loss of CO2 (characteristic of Curtius rearrangement)
        3. Formation of aniline if purpose is aniline_formation
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
                
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            
            # Filter out None molecules
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            if not reactants or not products:
                return False
                
            # Check for carboxylic acid in reactants
            carboxylic_acid_pattern = Chem.MolFromSmarts("[C](=O)[OH]")
            has_carboxylic_acid = any(mol.HasSubstructMatch(carboxylic_acid_pattern) for mol in reactants)
            
            # Check for primary amine in products
            primary_amine_pattern = Chem.MolFromSmarts("[NH2]")
            has_primary_amine = any(mol.HasSubstructMatch(primary_amine_pattern) for mol in products)
            
            # Check for CO2 loss (CO2 should appear in products as byproduct)
            co2_pattern = Chem.MolFromSmarts("C(=O)=O")
            has_co2_loss = any(mol.HasSubstructMatch(co2_pattern) for mol in products)
            
            # Basic Curtius rearrangement pattern
            is_curtius = has_carboxylic_acid and has_primary_amine and has_co2_loss
            
            # Additional check for aniline formation if specified
            if self.purpose == "aniline_formation" and is_curtius:
                aniline_pattern = Chem.MolFromSmarts("c1ccccc1[NH2]")
                has_aniline = any(mol.HasSubstructMatch(aniline_pattern) for mol in products)
                return has_aniline
                
            return is_curtius
            
        except Exception:
            return False
