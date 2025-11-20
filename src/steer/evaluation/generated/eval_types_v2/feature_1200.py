"""Generated evaluation code for: Late stage carbamate formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageCarbamate(BaseScoring):
    """
    Evaluates synthesis routes for late-stage carbamate formation.
    Checks if carbamate formation occurs within the specified depth threshold.
    """
    
    def __init__(self, config: Dict):
        self.depth_threshold = config.get("depth_threshold", 2)
        self.timing = config.get("timing", "late")
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Carbamate formation doesn't happen
        
        # For late-stage timing, reward reactions at shallow depths
        if self.timing == "late":
            if x <= self.depth_threshold / 10.0:  # Convert to fraction
                return 10  # Perfect score for very late stage
            else:
                return max(0, 10 - (x * 50))  # Penalty for earlier stages
        else:
            # General case - later is better
            return 10 * (1 - x)
    
    def hit_condition(self, d) -> bool:
        """
        Detects carbamate formation by identifying the characteristic 
        carbamate linkage (R-NH-CO-O-R') formation in the reaction.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn:
            return False
        
        try:
            rxn_parts = mapped_rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            products = rxn_parts[0]
            reactants = rxn_parts[1]
            
            # Parse molecules
            product_mol = Chem.MolFromSmiles(products)
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            
            if not product_mol or not all(reactant_mols):
                return False
            
            # Define carbamate pattern: R-NH-CO-O-R'
            carbamate_pattern = Chem.MolFromSmarts("[#6,#1][NH1][C](=[O])[O][#6]")
            
            # Check if product contains carbamate and reactants don't
            product_has_carbamate = product_mol.HasSubstructMatch(carbamate_pattern)
            reactants_have_carbamate = any(mol.HasSubstructMatch(carbamate_pattern) 
                                         for mol in reactant_mols if mol)
            
            # Carbamate formation: product has it, reactants don't
            if product_has_carbamate and not reactants_have_carbamate:
                return True
            
            # Additional check for specific carbamate formation patterns
            # Look for acyl chloride + amine/alcohol reactions
            acyl_chloride_pattern = Chem.MolFromSmarts("[C](=[O])[Cl]")
            amine_pattern = Chem.MolFromSmarts("[NH2,NH1]")
            alcohol_pattern = Chem.MolFromSmarts("[OH1]")
            
            has_acyl_chloride = any(mol.HasSubstructMatch(acyl_chloride_pattern) 
                                  for mol in reactant_mols if mol)
            has_amine = any(mol.HasSubstructMatch(amine_pattern) 
                           for mol in reactant_mols if mol)
            has_alcohol = any(mol.HasSubstructMatch(alcohol_pattern) 
                             for mol in reactant_mols if mol)
            
            # Check for carbamate formation via acyl chloride route
            if (product_has_carbamate and has_acyl_chloride and 
                (has_amine or has_alcohol)):
                return True
                
        except Exception:
            return False
            
        return False
