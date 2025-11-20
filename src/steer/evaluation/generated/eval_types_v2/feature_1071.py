"""Generated evaluation code for: Convergent synthesis via fragment coupling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentStrategy(BaseScoring):
    """
    Evaluates convergent synthesis strategy by detecting fragment coupling reactions.
    Checks if a specific coupling reaction type occurs at or before a target depth,
    indicating that complex fragments were synthesized separately and then joined.
    """
    
    def __init__(self, config: Dict):
        self.target_depth = config["coupling_reaction_depth"]
        self.coupling_type = config["coupling_reaction_type"]
        
        # Define SMARTS patterns for different coupling reactions
        self.coupling_patterns = {
            "esterification": {
                "bond_formed": "[C:1](=[O:2])[O:3][C:4]",  # Ester bond
                "reactant1": "[C:1](=[O:2])[OH]",          # Carboxylic acid
                "reactant2": "[OH][C:4]"                   # Alcohol
            },
            "amidation": {
                "bond_formed": "[C:1](=[O:2])[N:3][C:4]",  # Amide bond
                "reactant1": "[C:1](=[O:2])[OH]",          # Carboxylic acid
                "reactant2": "[N:3][C:4]"                  # Amine
            },
            "suzuki_coupling": {
                "bond_formed": "[c:1][c:2]",               # Aryl-aryl bond
                "reactant1": "[c:1][B]",                   # Boronic acid/ester
                "reactant2": "[c:2][Br,I,Cl]"              # Aryl halide
            },
            "click_chemistry": {
                "bond_formed": "[c:1]1[n:2][n:3][n:4][c:5]1", # Triazole ring
                "reactant1": "[C:1]#[C]",                  # Alkyne
                "reactant2": "[C:5][N:2]=[N+]=[N-]"        # Azide
            }
        }
    
    def route_scoring(self, x) -> float:
        """Convert depth fraction to score. Earlier coupling (lower depth) scores higher."""
        if x < 0:
            return 0  # Coupling reaction not found
        
        # Score based on how early the coupling occurs
        # Earlier coupling (x closer to 0) gets higher score
        return max(0, 10 * (1 - x))
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction node represents the target coupling reaction."""
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            return self._is_coupling_reaction(rxn_smiles)
        except KeyError:
            return False
    
    def _is_coupling_reaction(self, rxn_smiles: str) -> bool:
        """Determine if the reaction is the specified coupling type."""
        if self.coupling_type not in self.coupling_patterns:
            return False
        
        patterns = self.coupling_patterns[self.coupling_type]
        
        try:
            reactants_smiles, product_smiles = rxn_smiles.split(">>")
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check if product contains the characteristic bond/structure
            bond_pattern = Chem.MolFromSmarts(patterns["bond_formed"])
            if not product.HasSubstructMatch(bond_pattern):
                return False
            
            # Check if reactants contain the expected functional groups
            reactant1_pattern = Chem.MolFromSmarts(patterns["reactant1"])
            reactant2_pattern = Chem.MolFromSmarts(patterns["reactant2"])
            
            has_reactant1 = any(r.HasSubstructMatch(reactant1_pattern) for r in reactants)
            has_reactant2 = any(r.HasSubstructMatch(reactant2_pattern) for r in reactants)
            
            # Additional check: ensure this is actually a fragment coupling
            # (both reactants should be reasonably complex, not simple building blocks)
            complex_reactants = [r for r in reactants if r.GetNumHeavyAtoms() >= 6]
            
            return has_reactant1 and has_reactant2 and len(complex_reactants) >= 2
            
        except Exception:
            return False
