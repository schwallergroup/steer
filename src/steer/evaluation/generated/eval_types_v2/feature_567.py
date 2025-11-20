"""Generated evaluation code for: Curtius rearrangement for amine installation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class CurtiusRearrangement(BaseScoring):
    """
    Evaluates synthesis routes for the presence of Curtius rearrangement reactions.
    
    The Curtius rearrangement converts carboxylic acids to amines via acyl azide 
    intermediates, typically resulting in Boc-protected amines. This class detects
    the characteristic transformation pattern where a carboxylic acid is converted
    to an amine with loss of CO2.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "bool")
        self.target_depth = config.get("target_depth", {}).get("value", -1)
    
    def route_scoring(self, x) -> float:
        """Convert depth fraction to score (0-10 scale)"""
        if self.condition_type == "bool":
            if self.target_depth == -1:  # Positive if condition not met
                return 1 if x < 0 else 0
        else:
            if x < 0:
                return 0
            return abs(x - self.target_depth)
    
    def hit_condition(self, d) -> bool:
        """Check if a reaction node represents a Curtius rearrangement"""
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
            carboxylic_acid_pattern = Chem.MolFromSmarts("[CX3](=O)[OH]")
            has_carboxylic_acid = any(mol.HasSubstructMatch(carboxylic_acid_pattern) for mol in reactants)
            
            # Check for amine formation in products (including Boc-protected amines)
            amine_patterns = [
                Chem.MolFromSmarts("[NX3;H2,H1;!$(NC=O)]"),  # Primary/secondary amine
                Chem.MolFromSmarts("[NX3]C(=O)OC(C)(C)C"),    # Boc-protected amine
                Chem.MolFromSmarts("[NX3]C(=O)O[CH2]c1ccccc1") # Cbz-protected amine
            ]
            
            has_amine_product = any(
                any(mol.HasSubstructMatch(pattern) for pattern in amine_patterns)
                for mol in products
            )
            
            # Check for characteristic reagents that indicate Curtius conditions
            curtius_reagents = [
                Chem.MolFromSmarts("[N-]=[N+]=[N-]"),  # Azide ion
                Chem.MolFromSmarts("ClC(=O)OCC"),       # Ethyl chloroformate
                Chem.MolFromSmarts("N=[N+]=[N-]"),      # Sodium azide
                Chem.MolFromSmarts("CC(C)(C)OC(=O)Cl"), # Boc anhydride/Boc-Cl
            ]
            
            has_curtius_reagent = any(
                any(mol.HasSubstructMatch(reagent) for reagent in curtius_reagents)
                for mol in reactants
            )
            
            # Additional check for CO2 loss (characteristic of Curtius rearrangement)
            reactant_heavy_atoms = sum(mol.GetNumHeavyAtoms() for mol in reactants)
            product_heavy_atoms = sum(mol.GetNumHeavyAtoms() for mol in products)
            
            # Account for CO2 loss and possible reagent consumption
            # This is a rough heuristic as exact atom mapping would be more precise
            potential_co2_loss = reactant_heavy_atoms > product_heavy_atoms
            
            # Curtius rearrangement identified if:
            # 1. Carboxylic acid starting material present
            # 2. Amine product formed (protected or unprotected)
            # 3. Characteristic reagents present OR evidence of CO2 loss
            return (has_carboxylic_acid and has_amine_product and 
                   (has_curtius_reagent or potential_co2_loss))
                   
        except Exception:
            return False
