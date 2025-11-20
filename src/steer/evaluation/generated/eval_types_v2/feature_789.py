"""Generated evaluation code for: Terminal Williamson ether synthesis final step"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class TerminalWilliamsonEther(BaseScoring):
    """
    Evaluates whether the final step of a synthesis route is a Williamson ether synthesis.
    This checks for C-O bond formation via nucleophilic substitution of an alkoxide with an alkyl halide.
    """
    
    def __init__(self, config: Dict):
        self.position = config["parameters"].get("position", "final_step")
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Condition not met
        if self.position == "final_step":
            # For final step, we want x to be close to 1 (last step)
            if x > 0.9:  # Final step (depth fraction close to 1)
                return 10
            else:
                return 0
        else:
            # For other positions, score based on when it occurs
            return 10 * (1 - abs(x - 0.5))  # Prefer middle of route
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction is a Williamson ether synthesis"""
        metadata = d.get("metadata", {})
        rxn_smiles = metadata.get("mapped_reaction_smiles", "")
        
        if not rxn_smiles or ">>" not in rxn_smiles:
            return False
            
        try:
            reactants_smiles, product_smiles = rxn_smiles.split(">>")
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
                
            # Check for C-O ether bond formation
            if not self._has_ether_formation(reactants, product):
                return False
                
            # Check for Williamson ether synthesis pattern
            return self._is_williamson_mechanism(reactants)
            
        except Exception:
            return False
    
    def _has_ether_formation(self, reactants, product) -> bool:
        """Check if a new C-O ether bond is formed"""
        # Count ether oxygens in product (C-O-C pattern)
        ether_pattern = Chem.MolFromSmarts("[C]-[O]-[C]")
        product_ethers = len(product.GetSubstructMatches(ether_pattern))
        
        # Count ether oxygens in all reactants
        reactant_ethers = sum(len(r.GetSubstructMatches(ether_pattern)) for r in reactants)
        
        # New ether bond formed if product has more than reactants
        return product_ethers > reactant_ethers
    
    def _is_williamson_mechanism(self, reactants) -> bool:
        """Check for typical Williamson ether synthesis reactants"""
        has_alkoxide = False
        has_alkyl_halide = False
        
        # Patterns for alkoxide (R-O- or R-O-M where M is metal)
        alkoxide_patterns = [
            "[C]-[O-]",  # Alkoxide anion
            "[C]-[O]-[Na,K,Li,Mg,Ca]",  # Metal alkoxide
            "[C]-[OH]"  # Alcohol (can be converted to alkoxide in situ)
        ]
        
        # Patterns for alkyl halides
        halide_patterns = [
            "[C]-[Cl,Br,I]",  # Primary/secondary alkyl halides
            "[CH2]-[Cl,Br,I]",  # Primary alkyl halides (preferred)
        ]
        
        for reactant in reactants:
            # Check for alkoxide
            for pattern_smarts in alkoxide_patterns:
                pattern = Chem.MolFromSmarts(pattern_smarts)
                if pattern and reactant.HasSubstructMatch(pattern):
                    has_alkoxide = True
                    break
            
            # Check for alkyl halide
            for pattern_smarts in halide_patterns:
                pattern = Chem.MolFromSmarts(pattern_smarts)
                if pattern and reactant.HasSubstructMatch(pattern):
                    has_alkyl_halide = True
                    break
        
        return has_alkoxide and has_alkyl_halide
