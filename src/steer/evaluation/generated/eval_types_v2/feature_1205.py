"""Generated evaluation code for: Late stage carbamate formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageCarbamate(BaseScoring):
    """
    Evaluates whether carbamate formation occurs late in the synthesis route.
    Detects carbamate formation by looking for the presence of carbamate functional groups
    and common carbamate formation patterns in reactions.
    """
    
    def __init__(self, config: Dict):
        self.timing = config.get("timing", "late")
    
    def route_scoring(self, x) -> float:
        """
        Convert depth fraction to score where late-stage reactions get higher scores.
        """
        if x < 0:
            return 0  # Carbamate formation doesn't happen
        else:
            return 1 - x  # Later stage gets better score (closer to 1.0)
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves carbamate formation.
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles:
                return False
                
            rxn_parts = rxn_smiles.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants = rxn_parts[0]
            products = rxn_parts[1].split(".")[0]  # Take first product
            
            # Parse molecules
            reactant_mols = []
            for r_smiles in reactants.split("."):
                mol = Chem.MolFromSmiles(r_smiles)
                if mol:
                    reactant_mols.append(mol)
            
            product_mol = Chem.MolFromSmiles(products)
            if not product_mol:
                return False
            
            # Define carbamate pattern: N-C(=O)-O
            carbamate_pattern = Chem.MolFromSmarts("[NX3]-C(=O)-[OX2]")
            
            # Check if product has carbamate and reactants don't have it in same position
            product_has_carbamate = product_mol.HasSubstructMatch(carbamate_pattern)
            
            if not product_has_carbamate:
                return False
            
            # Check for typical carbamate formation patterns
            # Pattern 1: Carbamoyl chloride + alcohol/amine
            carbamoyl_chloride = Chem.MolFromSmarts("N-C(=O)-Cl")
            alcohol_pattern = Chem.MolFromSmarts("[OH1]")
            
            # Pattern 2: Isocyanate + alcohol
            isocyanate_pattern = Chem.MolFromSmarts("N=C=O")
            
            # Pattern 3: Chloroformate + amine
            chloroformate_pattern = Chem.MolFromSmarts("Cl-C(=O)-O")
            amine_pattern = Chem.MolFromSmarts("[NX3H2,NX3H1]")
            
            has_carbamoyl_chloride = any(mol.HasSubstructMatch(carbamoyl_chloride) for mol in reactant_mols)
            has_alcohol = any(mol.HasSubstructMatch(alcohol_pattern) for mol in reactant_mols)
            has_isocyanate = any(mol.HasSubstructMatch(isocyanate_pattern) for mol in reactant_mols)
            has_chloroformate = any(mol.HasSubstructMatch(chloroformate_pattern) for mol in reactant_mols)
            has_amine = any(mol.HasSubstructMatch(amine_pattern) for mol in reactant_mols)
            
            # Check if any typical carbamate formation pattern is present
            pattern1 = has_carbamoyl_chloride and has_alcohol
            pattern2 = has_isocyanate and has_alcohol
            pattern3 = has_chloroformate and has_amine
            
            return pattern1 or pattern2 or pattern3
            
        except Exception:
            return False
