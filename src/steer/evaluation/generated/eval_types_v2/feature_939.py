"""Generated evaluation code for: Late purine ring formation on sugar scaffold"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LatePurineRingFormation(BaseScoring):
    """
    Evaluates routes for late-stage purine ring formation on sugar scaffolds.
    Rewards routes where purine rings are formed directly on pre-existing sugar
    scaffolds rather than coupling pre-formed purine components.
    """
    
    def __init__(self, config: Dict):
        self.purine_smarts = config["parameters"]["ring_smarts"]  # "c1ncnc2[nH]cnc12"
        self.sugar_smarts = config["parameters"]["substrate_contains"]  # "C1OC(CO)CC1"
        self.timing = config["parameters"]["timing"]  # "late"
        
        # Compile patterns for efficiency
        self.purine_pattern = Chem.MolFromSmarts(self.purine_smarts)
        self.sugar_pattern = Chem.MolFromSmarts(self.sugar_smarts)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Condition not met
        
        if self.timing == "late":
            # Later formation is better (closer to 1.0 = later)
            return 10 * (1 - x)
        else:
            return 10 * x
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction forms a purine ring on a sugar-containing substrate.
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
                
            reactants_smiles, product_smiles = rxn_smiles.split(">>")
            
            # Parse molecules
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check if product contains both purine and sugar
            has_purine_product = product.HasSubstructMatch(self.purine_pattern)
            has_sugar_product = product.HasSubstructMatch(self.sugar_pattern)
            
            if not (has_purine_product and has_sugar_product):
                return False
            
            # Check that at least one reactant has sugar but no purine (scaffold present)
            scaffold_present = False
            purine_formed = True  # Assume purine is formed unless found in reactants
            
            for reactant in reactants:
                has_sugar_reactant = reactant.HasSubstructMatch(self.sugar_pattern)
                has_purine_reactant = reactant.HasSubstructMatch(self.purine_pattern)
                
                if has_sugar_reactant and not has_purine_reactant:
                    scaffold_present = True
                    
                if has_purine_reactant:
                    purine_formed = False  # Purine was already present
            
            # True if sugar scaffold was present and purine ring was formed in this step
            return scaffold_present and purine_formed
            
        except Exception:
            return False
