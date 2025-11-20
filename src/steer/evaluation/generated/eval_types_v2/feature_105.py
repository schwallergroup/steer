"""Generated evaluation code for: Convergent amide coupling strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentAmideCoupling(BaseScoring):
    """
    Evaluates convergent amide coupling strategy where two major fragments 
    are joined via amide bond formation.
    """
    
    def __init__(self, config: Dict):
        self.fragment_count = config["parameters"].get("fragment_count", 2)
        # SMARTS pattern for amide bond formation (C(=O)N)
        self.amide_pattern = Chem.MolFromSmarts("[C](=[O])[N]")
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Amide coupling doesn't happen
        else:
            # Reward earlier convergent coupling (lower depth)
            # Scale from 0-10 where earlier is better
            return max(0, 10 * (1 - x))
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction represents an amide coupling between major fragments.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            product = Chem.MolFromSmiles(rxn_parts[0])
            reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[1].split(".")]
            
            if not product or len(reactants) < self.fragment_count:
                return False
                
            # Check if product contains amide bond
            if not product.HasSubstructMatch(self.amide_pattern):
                return False
                
            # Check if this is a convergent coupling:
            # 1. Product should have significantly more atoms than individual reactants
            # 2. Reactants should be reasonably sized fragments (not small reagents)
            product_atom_count = product.GetNumAtoms()
            major_reactants = [r for r in reactants if r.GetNumAtoms() > 5]  # Filter out small reagents
            
            if len(major_reactants) < self.fragment_count:
                return False
                
            # Check if amide bond was actually formed in this step
            # by verifying reactants don't contain the same amide connectivity
            amide_formed = True
            for reactant in major_reactants:
                # If any major reactant already contains the full amide pattern,
                # this might not be the coupling step
                if reactant.HasSubstructMatch(self.amide_pattern):
                    # Additional check: see if the amide atoms are connected to both fragments
                    matches = reactant.GetSubstructMatches(self.amide_pattern)
                    if len(matches) > 0:
                        # This reactant already has amide - not the coupling step
                        amide_formed = False
                        break
            
            # Verify this is a convergent step by checking atom count ratios
            total_reactant_atoms = sum(r.GetNumAtoms() for r in major_reactants)
            convergent_ratio = product_atom_count / total_reactant_atoms if total_reactant_atoms > 0 else 0
            
            # Should be close to 1.0 for true convergent coupling (minus small leaving groups)
            return amide_formed and 0.8 <= convergent_ratio <= 1.2
            
        except Exception:
            return False
