"""Generated evaluation code for: Late stage trichloroacetimidate ether formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class TrichloroacetimidateEtherFormation(BaseScoring):
    """
    Evaluates synthesis routes for late-stage trichloroacetimidate ether formation.
    Detects the formation of C-O bonds using trichloroacetimidate activation,
    favoring routes where this reaction occurs late in the synthesis.
    """
    
    def __init__(self, config: Dict):
        self.timing_preference = config.get("timing", "late")
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Reaction doesn't occur in route
        
        if self.timing_preference == "late":
            return 1 - x  # Later stages get higher scores (closer to 1)
        else:
            return x  # Earlier stages get higher scores
    
    def hit_condition(self, d) -> bool:
        """
        Detects trichloroacetimidate ether formation by looking for:
        1. Trichloroacetimidate leaving group (CCl3-C(=N)-O-)
        2. C-O bond formation pattern
        3. Loss of the imidate group
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            reactants_smiles, products_smiles = mapped_rxn.split(">>")
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            if None in reactants or None in products:
                return False
            
            # Check for trichloroacetimidate pattern in reactants
            trichloroacetimidate_pattern = Chem.MolFromSmarts("[C](=[N])[O][#6]")  # Imidate ester
            trichlorocarbonyl_pattern = Chem.MolFromSmarts("C(Cl)(Cl)(Cl)C(=N)")  # CCl3-C=N
            
            has_imidate_reactant = False
            has_trichlorocarbonyl = False
            
            for reactant in reactants:
                if reactant.HasSubstructMatch(trichloroacetimidate_pattern):
                    has_imidate_reactant = True
                if reactant.HasSubstructMatch(trichlorocarbonyl_pattern):
                    has_trichlorocarbonyl = True
            
            if not (has_imidate_reactant and has_trichlorocarbonyl):
                return False
            
            # Check for new C-O bond formation
            # Look for alcohol reactant and ether product
            alcohol_pattern = Chem.MolFromSmarts("[C][OH]")
            ether_pattern = Chem.MolFromSmarts("[C][O][C]")
            
            has_alcohol_reactant = any(r.HasSubstructMatch(alcohol_pattern) for r in reactants)
            has_ether_product = any(p.HasSubstructMatch(ether_pattern) for p in products)
            
            # Check for trichloroacetamide byproduct (CCl3CONH2 or similar)
            amide_byproduct_pattern = Chem.MolFromSmarts("C(Cl)(Cl)(Cl)C(=O)[N]")
            has_amide_byproduct = any(p.HasSubstructMatch(amide_byproduct_pattern) for p in products)
            
            return has_alcohol_reactant and has_ether_product and has_amide_byproduct
            
        except Exception:
            return False
