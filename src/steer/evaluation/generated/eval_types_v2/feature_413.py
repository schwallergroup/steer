"""Generated evaluation code for: Late stage ether formation via Williamson reaction"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageWilliamsonEther(BaseScoring):
    """
    Evaluates synthesis routes for late-stage Williamson ether synthesis reactions.
    
    The Williamson ether synthesis involves nucleophilic substitution between an alkoxide
    and an alkyl halide/tosylate to form an ether bond. This class checks if such a
    reaction occurs in the later stages of synthesis (after position_threshold fraction
    of the route depth).
    """
    
    def __init__(self, config: Dict):
        self.position_threshold = config.get("position_threshold", 0.8)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Williamson reaction doesn't occur
        elif x >= self.position_threshold:
            return 10  # Perfect score for late-stage occurrence
        else:
            # Linear scaling - later is better
            return 10 * (x / self.position_threshold)
    
    def hit_condition(self, d) -> bool:
        """
        Detect Williamson ether synthesis by looking for:
        1. Formation of new C-O-C ether bond
        2. Presence of leaving group (halide, tosylate) in reactants
        3. Alkoxide or phenoxide nucleophile characteristics
        """
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            reactants_smiles, product_smiles = rxn_smiles.split(">>")
            
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check for ether bond formation in product
            ether_pattern = Chem.MolFromSmarts("[C,c]-O-[C,c]")
            if not product.HasSubstructMatch(ether_pattern):
                return False
            
            # Look for leaving groups in reactants (halides, tosylates, mesylates)
            leaving_groups = [
                "[C,c][Cl,Br,I]",  # Alkyl/aryl halides
                "[C,c]OS(=O)(=O)[c]",  # Tosylates
                "[C,c]OS(=O)(=O)[C]"   # Mesylates
            ]
            
            has_leaving_group = False
            for lg_smarts in leaving_groups:
                lg_pattern = Chem.MolFromSmarts(lg_smarts)
                if any(reactant.HasSubstructMatch(lg_pattern) for reactant in reactants):
                    has_leaving_group = True
                    break
            
            if not has_leaving_group:
                return False
            
            # Check for alkoxide/phenoxide nucleophile indicators
            # Look for oxygen-containing reactants that could be nucleophiles
            nucleophile_patterns = [
                "[O-]",  # Alkoxide anion
                "[c]O",  # Phenol (can form phenoxide)
                "[C]O"   # Alcohol (can form alkoxide)
            ]
            
            has_nucleophile = False
            for nuc_smarts in nucleophile_patterns:
                nuc_pattern = Chem.MolFromSmarts(nuc_smarts)
                if any(reactant.HasSubstructMatch(nuc_pattern) for reactant in reactants):
                    has_nucleophile = True
                    break
            
            return has_nucleophile and has_leaving_group
            
        except Exception:
            return False
