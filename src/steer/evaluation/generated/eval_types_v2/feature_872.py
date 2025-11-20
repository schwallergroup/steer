"""Generated evaluation code for: Acetate protecting group for cross-coupling compatibility"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class AcetateProtectingGroupStrategy(BaseScoring):
    """
    Evaluates synthesis routes for the use of acetate protecting groups on primary alcohols
    to enable cross-coupling compatibility. Returns higher scores when acetate protection
    occurs earlier in the synthesis.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "continuous")
        self.target_depth = config.get("target_depth", {}).get("value", 0.3)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Protection strategy not used
        else:
            # Earlier protection is better for compatibility
            return max(0, 10 * (1 - x))
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves acetate protection of a primary alcohol
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
        
        try:
            reactants_smiles, products_smiles = mapped_rxn.split(">>")
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            # Filter out None molecules
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            if not reactants or not products:
                return False
            
            # Check for acetate formation: primary alcohol + acetyl source -> acetate ester
            primary_alcohol_pattern = Chem.MolFromSmarts("[CH2][OH]")
            acetate_ester_pattern = Chem.MolFromSmarts("[CH2]OC(=O)C")
            acetyl_source_pattern = Chem.MolFromSmarts("C(=O)C")  # Acetyl group source
            
            # Check if reactants contain primary alcohol and acetyl source
            has_primary_alcohol = any(mol.HasSubstructMatch(primary_alcohol_pattern) for mol in reactants)
            has_acetyl_source = any(mol.HasSubstructMatch(acetyl_source_pattern) for mol in reactants)
            
            # Check if products contain acetate ester
            has_acetate_ester = any(mol.HasSubstructMatch(acetate_ester_pattern) for mol in products)
            
            # Also check for common acetylating reagents
            acetylating_reagents = [
                "CC(=O)Cl",  # Acetyl chloride
                "CC(=O)OC(=O)C",  # Acetic anhydride
                "CC(=O)O"  # Acetic acid (with coupling agent)
            ]
            
            has_acetylating_reagent = any(
                any(Chem.MolFromSmiles(reagent).HasSubstructMatch(mol) or 
                    mol.HasSubstructMatch(Chem.MolFromSmiles(reagent)) 
                    for mol in reactants if mol is not None)
                for reagent in acetylating_reagents
            )
            
            return (has_primary_alcohol and has_acetate_ester and 
                   (has_acetyl_source or has_acetylating_reagent))
            
        except Exception:
            return False
