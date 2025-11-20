"""Generated evaluation code for: Early stage ketal protection strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlyKetालProtection(BaseScoring):
    """
    Evaluates if ketal protection of ketones occurs in the early stages of synthesis.
    Returns higher scores when ketal protection happens earlier in the route.
    """
    
    def __init__(self, config: Dict):
        self.timing_preference = config.get("timing", "early")
        self.protection_type = config.get("protection_type", "ketal")
        
    def route_scoring(self, x) -> float:
        """
        Score based on when ketal protection occurs.
        Early protection gets higher scores.
        """
        if x < 0:
            return 0  # No ketal protection found
        
        # For early timing preference, earlier reactions get higher scores
        if self.timing_preference == "early":
            return 1 - x  # x is depth fraction, so 1-x rewards early reactions
        else:
            return x  # Later reactions get higher scores
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves ketal protection of a ketone.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        reactants_smiles, products_smiles = mapped_rxn.split(">>")
        
        try:
            # Parse reactants and products
            reactant_mols = [Chem.MolFromSmiles(smi.strip()) 
                           for smi in reactants_smiles.split(".")]
            product_mols = [Chem.MolFromSmiles(smi.strip()) 
                          for smi in products_smiles.split(".")]
            
            if not all(reactant_mols) or not all(product_mols):
                return False
            
            # Define ketone pattern (C=O)
            ketone_pattern = Chem.MolFromSmarts("[CX3]=[OX1]")
            
            # Define ketal patterns (common dimethyl ketal and other acetals/ketals)
            ketal_patterns = [
                Chem.MolFromSmarts("[CX4]([OX2])([OX2])[CX4]"),  # General ketal pattern
                Chem.MolFromSmarts("CO[CX4]OC"),  # Dimethyl ketal
                Chem.MolFromSmarts("[CH2][OX2][CX4][OX2][CH2]"),  # Ethylene glycol ketal
            ]
            
            # Check if reactants have ketones
            has_ketone_in_reactants = any(
                mol.HasSubstructMatch(ketone_pattern) for mol in reactant_mols
            )
            
            # Check if products have ketals
            has_ketal_in_products = any(
                any(mol.HasSubstructMatch(pattern) for pattern in ketal_patterns)
                for mol in product_mols
            )
            
            # Also check for ketal-forming reagents (alcohols like methanol, ethylene glycol)
            ketal_reagents = [
                Chem.MolFromSmarts("CO"),  # Methanol
                Chem.MolFromSmarts("CCO"),  # Ethanol  
                Chem.MolFromSmarts("OCCO"),  # Ethylene glycol
            ]
            
            has_ketal_reagent = any(
                any(mol.HasSubstructMatch(reagent) for reagent in ketal_reagents)
                for mol in reactant_mols
            )
            
            # Ketal protection: ketone + alcohol reagent -> ketal product
            return (has_ketone_in_reactants and 
                   has_ketal_reagent and 
                   has_ketal_in_products)
            
        except Exception:
            return False
