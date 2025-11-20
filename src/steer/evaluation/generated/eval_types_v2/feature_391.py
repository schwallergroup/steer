"""Generated evaluation code for: Early azide installation for nitrogen introduction"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlyAzideInstallation(BaseScoring):
    """
    Evaluates whether azide installation occurs early in the synthesis route.
    
    Detects azide substitution reactions (formation of R-N3 groups) and scores
    based on how early this occurs, with step 1 being optimal for nitrogen
    precursor installation.
    """
    
    def __init__(self, config: Dict):
        self.target_step = config["parameters"]["step_position"]
        self.total_steps = config["parameters"]["total_steps"]
    
    def route_scoring(self, x) -> float:
        """
        Score based on how close the azide installation is to the target early step.
        Early installation (step 1-2) gets high scores, later steps get lower scores.
        """
        if x < 0:
            return 0  # No azide installation found
        
        # Convert depth fraction to actual step number
        actual_step = int(x * self.total_steps) + 1
        
        # Calculate deviation from target step
        step_deviation = abs(actual_step - self.target_step)
        
        # Score decreases with deviation, with maximum penalty at late steps
        if step_deviation == 0:
            return 10  # Perfect early installation
        elif actual_step <= 3:
            return max(8 - step_deviation, 6)  # Still good if within first 3 steps
        elif actual_step <= 5:
            return max(6 - step_deviation, 3)  # Moderate score for mid-early steps
        else:
            return max(3 - (step_deviation * 0.5), 0)  # Low score for late installation
    
    def hit_condition(self, d) -> bool:
        """
        Detect azide substitution reactions by checking for azide group formation.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
        
        reactants_smiles, products_smiles = mapped_rxn.split(">>")
        
        try:
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            if None in reactants or None in products:
                return False
            
            # Define azide patterns
            azide_pattern = Chem.MolFromSmarts("[N-]=[N+]=[N-]")  # Azide group N3-
            azide_alkyl_pattern = Chem.MolFromSmarts("C-[N-]=[N+]=[N-]")  # R-N3
            
            # Check if products contain azide groups that weren't in reactants
            reactant_has_azide = any(mol.HasSubstructMatch(azide_pattern) for mol in reactants if mol)
            product_has_azide = any(mol.HasSubstructMatch(azide_alkyl_pattern) for mol in products if mol)
            
            # Also check for azide ion as reactant (nucleophilic substitution)
            azide_ion_present = any("N=[N+]=[N-]" in Chem.MolToSmiles(mol) or 
                                  "[N-]=[N+]=[N-]" in Chem.MolToSmiles(mol) 
                                  for mol in reactants if mol)
            
            # Azide installation: azide ion + substrate -> azide product
            return (azide_ion_present and product_has_azide) or \
                   (not reactant_has_azide and product_has_azide)
            
        except Exception:
            return False
