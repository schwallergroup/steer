"""Generated evaluation code for: Late stage sulfonamide formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageSulfonamideFormation(BaseScoring):
    """
    Evaluates synthesis routes for late-stage sulfonamide formation.
    
    Detects sulfonamide bond formation reactions and rewards routes where
    this reaction occurs in the final steps of the synthesis.
    """
    
    def __init__(self, config: Dict):
        self.timing_preference = config.get("timing", "late")
        self.step_position = config.get("step_position", "final")
    
    def route_scoring(self, x) -> float:
        """
        Converts depth fraction to score (0-10).
        For late-stage preference, higher depth fractions get better scores.
        """
        if x < 0:
            return 0  # Sulfonamide formation doesn't happen
        
        if self.timing_preference == "late":
            # Reward late-stage sulfonamide formation (higher depth = better score)
            return 10 * x
        else:
            # For early timing preference
            return 10 * (1 - x)
    
    def hit_condition(self, d) -> bool:
        """
        Checks if a reaction node represents sulfonamide formation.
        
        Detects patterns where:
        1. Sulfonyl chloride reacts with amine to form sulfonamide
        2. General S-N bond formation in sulfonamide context
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            product_smiles, reactant_smiles = rxn_smiles.split(">>")
            product_mol = Chem.MolFromSmiles(product_smiles)
            reactant_mols = [Chem.MolFromSmiles(r) for r in reactant_smiles.split(".")]
            
            if not product_mol or not all(reactant_mols):
                return False
            
            # Check for sulfonamide formation patterns
            return self._is_sulfonamide_formation(product_mol, reactant_mols)
            
        except Exception:
            return False
    
    def _is_sulfonamide_formation(self, product_mol, reactant_mols) -> bool:
        """
        Detects sulfonamide formation by checking:
        1. Product contains sulfonamide group
        2. Reactants contain sulfonyl chloride and amine components
        """
        # Sulfonamide pattern: S(=O)(=O)N
        sulfonamide_pattern = Chem.MolFromSmarts("[#16](=[#8])(=[#8])[#7]")
        if not product_mol.HasSubstructMatch(sulfonamide_pattern):
            return False
        
        # Check reactants for typical sulfonamide formation components
        has_sulfonyl_chloride = False
        has_amine = False
        
        for reactant in reactant_mols:
            # Sulfonyl chloride pattern: S(=O)(=O)Cl
            sulfonyl_chloride_pattern = Chem.MolFromSmarts("[#16](=[#8])(=[#8])[#17]")
            if reactant.HasSubstructMatch(sulfonyl_chloride_pattern):
                has_sulfonyl_chloride = True
            
            # Amine patterns: primary or secondary amines
            primary_amine_pattern = Chem.MolFromSmarts("[#7;H2]")
            secondary_amine_pattern = Chem.MolFromSmarts("[#7;H1]")
            if (reactant.HasSubstructMatch(primary_amine_pattern) or 
                reactant.HasSubstructMatch(secondary_amine_pattern)):
                has_amine = True
        
        return has_sulfonyl_chloride and has_amine
