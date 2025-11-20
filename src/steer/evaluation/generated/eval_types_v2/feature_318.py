"""Generated evaluation code for: Convergent synthesis via two fragments"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentSynthesis(BaseScoring):
    """
    Evaluates convergent synthesis strategy by detecting when multiple fragments 
    are coupled together at a specific stage of the synthesis route.
    
    This class identifies coupling reactions where two or more substantial fragments
    are joined, particularly focusing on late-stage convergent steps.
    """
    
    def __init__(self, config: Dict):
        self.fragment_count = config.get("fragment_count", 2)
        self.coupling_stage = config.get("coupling_stage", "late")  # "early", "mid", "late"
        
        # Define minimum heavy atom count for a fragment to be considered substantial
        self.min_fragment_size = config.get("min_fragment_size", 8)
        
    def route_scoring(self, x) -> float:
        """Convert depth fraction to score based on coupling stage preference"""
        if x < 0:
            return 0  # No convergent coupling found
            
        if self.coupling_stage == "late":
            return 1 - x  # Prefer later coupling (lower depth fraction = higher score)
        elif self.coupling_stage == "early":
            return x  # Prefer earlier coupling (higher depth fraction = higher score)
        else:  # "mid"
            # Prefer middle stages, penalize very early or very late
            return 1 - abs(x - 0.5) * 2
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction represents a convergent coupling of multiple fragments
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            product_smiles = rxn_parts[0]
            reactants_smiles = rxn_parts[1]
            
            if "." not in reactants_smiles:
                return False  # Need multiple reactants for convergent synthesis
                
            # Parse molecules
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r) for r in reactants_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
                
            # Filter reactants to only substantial fragments
            substantial_reactants = [
                r for r in reactants 
                if r.GetNumHeavyAtoms() >= self.min_fragment_size
            ]
            
            # Check if we have the required number of substantial fragments
            if len(substantial_reactants) < self.fragment_count:
                return False
                
            # Verify this is actually a coupling reaction by checking that
            # significant portions of each reactant appear in the product
            return self._verify_coupling_reaction(product, substantial_reactants)
            
        except Exception:
            return False
    
    def _verify_coupling_reaction(self, product, reactants) -> bool:
        """
        Verify that this is a true coupling by checking that substantial
        portions of each reactant are preserved in the product
        """
        from rdkit.Chem import rdFMCS
        
        try:
            for reactant in reactants:
                # Find maximum common substructure between reactant and product
                mcs_result = rdFMCS.FindMCS([reactant, product], 
                                          bondCompare=rdFMCS.BondCompare.CompareAny,
                                          atomCompare=rdFMCS.AtomCompare.CompareAny,
                                          timeout=5)
                
                if mcs_result.numAtoms == 0:
                    return False
                    
                # Check that a significant portion of the reactant is preserved
                preservation_ratio = mcs_result.numAtoms / reactant.GetNumHeavyAtoms()
                if preservation_ratio < 0.6:  # At least 60% of reactant should be preserved
                    return False
                    
            return True
            
        except Exception:
            # Fallback: just check that we have multiple substantial reactants
            return len(reactants) >= self.fragment_count
