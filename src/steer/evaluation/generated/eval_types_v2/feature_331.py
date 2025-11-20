"""Generated evaluation code for: Late stage epoxide formation via Corey-Chaykovsky"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class CoreyChaykovsky(BaseScoring):
    """
    Evaluates synthesis routes for late-stage epoxide formation via Corey-Chaykovsky reaction.
    Checks for the presence of dimethylsulfoxonium ylide reagent and epoxide formation.
    """
    
    def __init__(self, config: Dict):
        # Dimethylsulfoxonium ylide pattern
        self.reagent_pattern = config.get("reagent_pattern", "C[S+](C)C")
        self.timing = config.get("timing", "late")
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Reaction doesn't occur
        else:
            # Late-stage reaction is better (lower depth fraction)
            return 1 - x
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction step involves Corey-Chaykovsky epoxidation.
        """
        try:
            rxn = d["metadata"]["mapped_reaction_smiles"].split(">>")
            product = Chem.MolFromSmiles(rxn[0])
            reactants = [Chem.MolFromSmiles(r) for r in rxn[1].split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check for dimethylsulfoxonium ylide reagent
            reagent_mol = Chem.MolFromSmarts(self.reagent_pattern)
            has_reagent = any(r.HasSubstructMatch(reagent_mol) for r in reactants)
            
            if not has_reagent:
                return False
            
            # Check for epoxide formation (3-membered ring with oxygen)
            epoxide_pattern = Chem.MolFromSmarts("C1OC1")
            product_has_epoxide = product.HasSubstructMatch(epoxide_pattern)
            
            # Check that reactants don't already have the epoxide
            reactants_have_epoxide = any(r.HasSubstructMatch(epoxide_pattern) for r in reactants)
            
            # Epoxide should be formed in this step
            return product_has_epoxide and not reactants_have_epoxide
            
        except Exception:
            return False
