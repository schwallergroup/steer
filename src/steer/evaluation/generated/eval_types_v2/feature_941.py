"""Generated evaluation code for: Late stage sulfone formation via thioether oxidation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageSulfoneFormation(BaseScoring):
    """
    Evaluates whether sulfone formation via thioether oxidation occurs in the final step.
    Detects the transformation of sulfide (C-S-C) to sulfone (C-SO2-C) patterns.
    """
    
    def __init__(self, config: Dict):
        self.position = config.get("parameters", {}).get("position", "final_step")
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Reaction doesn't occur
        
        if self.position == "final_step":
            # For final step, we want depth fraction close to 1.0 (very late)
            if x >= 0.8:  # Final step territory
                return 10.0
            else:
                return x * 5.0  # Partial credit for later stages
        else:
            # General late-stage preference
            return x * 10.0
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction node represents sulfide to sulfone oxidation."""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
        
        try:
            reactants_smiles, products_smiles = mapped_rxn.split(">>")
            
            # Parse reactants and products
            reactant_mols = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            product_mols = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            # Remove None molecules
            reactant_mols = [mol for mol in reactant_mols if mol is not None]
            product_mols = [mol for mol in product_mols if mol is not None]
            
            if not reactant_mols or not product_mols:
                return False
            
            # Define SMARTS patterns
            sulfide_pattern = Chem.MolFromSmarts("[C,c][S]([C,c])")  # C-S-C
            sulfone_pattern = Chem.MolFromSmarts("[C,c][S](=O)(=O)[C,c]")  # C-SO2-C
            
            if sulfide_pattern is None or sulfone_pattern is None:
                return False
            
            # Check for sulfide in reactants and sulfone in products
            has_sulfide_reactant = any(mol.HasSubstructMatch(sulfide_pattern) for mol in reactant_mols)
            has_sulfone_product = any(mol.HasSubstructMatch(sulfone_pattern) for mol in product_mols)
            
            # Additional check: ensure no sulfone in reactants (true oxidation)
            has_sulfone_reactant = any(mol.HasSubstructMatch(sulfone_pattern) for mol in reactant_mols)
            
            return has_sulfide_reactant and has_sulfone_product and not has_sulfone_reactant
            
        except Exception:
            return False
