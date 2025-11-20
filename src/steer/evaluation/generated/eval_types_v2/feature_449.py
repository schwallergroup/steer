"""Generated evaluation code for: Late purine ring formation via Traube synthesis"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LatePurineTraubeSynthesis(BaseScoring):
    """
    Evaluates routes for late-stage purine ring formation via Traube synthesis.
    
    Checks if the purine ring system (c1ncnc2[nH]cnc12) is formed late in the synthesis
    via Traube method, which involves cyclization of an ortho-aminonitrile precursor
    with formamide or similar reagents.
    """
    
    def __init__(self, config: Dict):
        self.purine_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]
        self.method = config["parameters"]["method"]
        
    def route_scoring(self, x) -> float:
        """
        Score based on depth of purine formation.
        Later formation (higher depth fraction) gets better score.
        """
        if x < 0:
            return 0  # Purine formation not detected
        else:
            # Late-stage formation is better - return depth fraction scaled to 0-10
            return x * 10
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction forms a purine ring via Traube synthesis.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        reactants_smiles, product_smiles = mapped_rxn.split(">>")
        
        try:
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
                
            # Check if product contains purine ring
            purine_pattern = Chem.MolFromSmarts(self.purine_smarts)
            if not product.HasSubstructMatch(purine_pattern):
                return False
                
            # Check if purine ring is newly formed (not present in reactants)
            purine_in_reactants = any(r.HasSubstructMatch(purine_pattern) for r in reactants if r)
            if purine_in_reactants:
                return False  # Purine already existed
                
            # Check for Traube synthesis pattern: ortho-aminonitrile precursor
            # Pattern: aminonitrile with adjacent amino and cyano groups
            traube_precursor_pattern = Chem.MolFromSmarts("[NH2]c1nc(C#N)nc[nH]1")  # Simplified traube precursor
            ortho_aminonitrile_pattern = Chem.MolFromSmarts("c1c(N)c(C#N)ccc1")  # General ortho-aminonitrile
            
            # Check if any reactant contains the precursor pattern
            has_traube_precursor = any(
                r.HasSubstructMatch(traube_precursor_pattern) or 
                r.HasSubstructMatch(ortho_aminonitrile_pattern)
                for r in reactants if r
            )
            
            # Check for formamide or similar cyclization reagents
            formamide_pattern = Chem.MolFromSmarts("NC=O")  # Formamide
            formic_acid_pattern = Chem.MolFromSmarts("C(=O)O")  # Formic acid
            
            has_cyclization_reagent = any(
                r.HasSubstructMatch(formamide_pattern) or
                r.HasSubstructMatch(formic_acid_pattern)
                for r in reactants if r
            )
            
            return has_traube_precursor and has_cyclization_reagent
            
        except Exception:
            return False
