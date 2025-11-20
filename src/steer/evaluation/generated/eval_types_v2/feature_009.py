"""Generated evaluation code for: Late stage Buchwald-Hartwig coupling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageBuchwaldHartwig(BaseScoring):
    """
    Evaluates if a Buchwald-Hartwig coupling reaction occurs at a late stage in the synthesis.
    Uses pattern matching to detect C-N bond formation between an aryl halide and an amine.
    Prefers reactions that occur within the specified depth threshold from the target.
    """
    
    def __init__(self, config: Dict):
        self.depth_threshold = config.get("depth_threshold", 2)
        # SMARTS patterns for Buchwald-Hartwig coupling detection
        self.aryl_halide_pattern = "[c,C][Cl,Br,I]"  # Aryl chloride/bromide/iodide
        self.amine_pattern = "[NH1,NH2]"  # Primary or secondary amine
        self.product_pattern = "[c,C][NH1,NH2]"  # Formed C-N bond
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Reaction doesn't occur
        
        # Convert depth fraction to score (0-1 range, then scale to 0-10)
        # Late stage means smaller depth fraction is better
        if x <= (self.depth_threshold / 10.0):  # Within threshold
            score = 1 - (x / (self.depth_threshold / 10.0)) * 0.2  # Small penalty for being later
        else:
            score = 0.8 * (1 - x)  # Larger penalty for being too early
        
        return max(0, min(1, score))
    
    def hit_condition(self, d) -> bool:
        """
        Check if a reaction node represents a Buchwald-Hartwig coupling.
        Looks for C-N bond formation from aryl halide + amine.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
        
        try:
            product_smiles, reactants_smiles = mapped_rxn.split(">>")
            product_mol = Chem.MolFromSmiles(product_smiles)
            reactant_mols = [Chem.MolFromSmiles(r.strip()) 
                           for r in reactants_smiles.split(".")]
            
            if not product_mol or not all(reactant_mols):
                return False
            
            # Check if product contains C-N bond pattern
            product_pattern_mol = Chem.MolFromSmarts(self.product_pattern)
            if not product_mol.HasSubstructMatch(product_pattern_mol):
                return False
            
            # Check if reactants contain aryl halide and amine
            has_aryl_halide = False
            has_amine = False
            
            aryl_halide_mol = Chem.MolFromSmarts(self.aryl_halide_pattern)
            amine_mol = Chem.MolFromSmarts(self.amine_pattern)
            
            for reactant in reactant_mols:
                if reactant.HasSubstructMatch(aryl_halide_mol):
                    has_aryl_halide = True
                if reactant.HasSubstructMatch(amine_mol):
                    has_amine = True
            
            # Additional check: ensure C-N bond is newly formed
            if has_aryl_halide and has_amine:
                # Verify none of the reactants already have the C-N pattern
                product_pattern_in_reactants = any(r.HasSubstructMatch(product_pattern_mol) 
                                                 for r in reactant_mols)
                return not product_pattern_in_reactants
            
            return False
            
        except Exception:
            return False
