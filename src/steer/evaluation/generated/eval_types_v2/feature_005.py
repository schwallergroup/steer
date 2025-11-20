"""Generated evaluation code for: Early quinoline core construction"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class QuinolineCoreConstruction(BaseScoring):
    """
    Evaluates when quinoline core construction occurs in synthesis routes.
    Rewards early formation of the quinoline heterocycle structure.
    """
    
    def __init__(self, config: Dict):
        self.quinoline_pattern = Chem.MolFromSmarts(config["parameters"]["ring_smarts"])
        self.timing = config["parameters"]["timing"]
        self.direction = config["parameters"]["direction"]
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Quinoline formation doesn't happen
        else:
            if self.timing == "early":
                return 1 - x  # Early formation is better (higher score for lower depth fraction)
            else:
                return x  # Late formation is better (higher score for higher depth fraction)
    
    def hit_condition(self, d) -> bool:
        """Check if quinoline core is formed in this reaction step."""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        rxn_parts = mapped_rxn.split(">>")
        product = rxn_parts[0]
        reactants = rxn_parts[1].split(".")
        
        try:
            prod_mol = Chem.MolFromSmiles(product)
            reactant_mols = [Chem.MolFromSmiles(r) for r in reactants if Chem.MolFromSmiles(r) is not None]
            
            if prod_mol is None or not reactant_mols:
                return False
            
            # Check if product contains quinoline pattern
            prod_has_quinoline = prod_mol.HasSubstructMatch(self.quinoline_pattern)
            
            if self.direction == "formation":
                # Check if any reactant lacks the quinoline pattern
                reactants_lack_quinoline = any(not r_mol.HasSubstructMatch(self.quinoline_pattern) 
                                             for r_mol in reactant_mols)
                return prod_has_quinoline and reactants_lack_quinoline
            else:  # breaking
                # Check if product lacks quinoline but reactants have it
                reactants_have_quinoline = any(r_mol.HasSubstructMatch(self.quinoline_pattern) 
                                             for r_mol in reactant_mols)
                return not prod_has_quinoline and reactants_have_quinoline
                
        except Exception:
            return False
