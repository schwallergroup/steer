"""Generated evaluation code for: Early benzylic bromide installation and retention"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BenzylicBromideRetention(BaseScoring):
    """
    Evaluates synthesis routes for early installation and retention of benzylic bromide groups.
    Checks if a benzylic C-Br bond is formed early and carried through multiple steps.
    """
    
    def __init__(self, config: Dict):
        self.min_steps_carried = config["parameters"].get("steps_carried", 8)
        self.timing_preference = config["parameters"].get("timing", "early")
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Benzylic bromide installation not detected
        
        if self.timing_preference == "early":
            # Earlier installation is better, penalize late installation
            return max(0, 1 - x)
        else:
            # Standard depth-based scoring
            return 1 - x
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction involves benzylic bromide installation and retention."""
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            rxn_parts = rxn_smiles.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            products = rxn_parts[0]
            reactants = rxn_parts[1]
            
            prod_mol = Chem.MolFromSmiles(products)
            reactant_mols = [Chem.MolFromSmiles(r) for r in reactants.split(".")]
            
            if not prod_mol or not all(reactant_mols):
                return False
            
            # Check if benzylic bromide is formed in this step
            benzylic_br_formed = self._has_benzylic_bromide_formation(prod_mol, reactant_mols)
            
            # Check if benzylic bromide is retained from previous steps
            benzylic_br_retained = self._has_benzylic_bromide_retention(prod_mol, reactant_mols)
            
            return benzylic_br_formed or benzylic_br_retained
            
        except Exception:
            return False
    
    def _has_benzylic_bromide_formation(self, product, reactants) -> bool:
        """Check if a benzylic bromide is formed in this reaction."""
        # Benzylic bromide pattern: aromatic carbon connected to CH2-Br or CHR-Br
        benzylic_br_pattern = Chem.MolFromSmarts("[cH0,cH1:1]-[CH1,CH2:2]-[Br:3]")
        
        if not benzylic_br_pattern:
            return False
        
        # Check if product has benzylic bromide
        prod_has_pattern = product.HasSubstructMatch(benzylic_br_pattern)
        
        if not prod_has_pattern:
            return False
        
        # Check if any reactant lacks this pattern (indicating formation)
        for reactant in reactants:
            if not reactant.HasSubstructMatch(benzylic_br_pattern):
                return True
        
        return False
    
    def _has_benzylic_bromide_retention(self, product, reactants) -> bool:
        """Check if a benzylic bromide is retained through this reaction."""
        benzylic_br_pattern = Chem.MolFromSmarts("[cH0,cH1:1]-[CH1,CH2:2]-[Br:3]")
        
        if not benzylic_br_pattern:
            return False
        
        # Check if both product and at least one reactant have the pattern
        prod_has_pattern = product.HasSubstructMatch(benzylic_br_pattern)
        reactant_has_pattern = any(r.HasSubstructMatch(benzylic_br_pattern) for r in reactants)
        
        return prod_has_pattern and reactant_has_pattern
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        """
        Override to check for multi-step retention of benzylic bromide.
        Returns True if benzylic bromide is carried through sufficient steps.
        """
        reactions = self.get_rxns(d)
        
        # Count consecutive steps where benzylic bromide is present
        consecutive_steps = 0
        max_consecutive = 0
        found_installation = False
        
        for reaction in reactions:
            if self.hit_condition(reaction):
                consecutive_steps += 1
                max_consecutive = max(max_consecutive, consecutive_steps)
                found_installation = True
            else:
                consecutive_steps = 0
        
        # Condition met if benzylic bromide found and carried for required steps
        condition_met = found_installation and max_consecutive >= self.min_steps_carried
        
        return condition_met, len(reactions)
