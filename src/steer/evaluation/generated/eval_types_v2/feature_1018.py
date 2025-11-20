"""Generated evaluation code for: Convergent synthesis via two fragment coupling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentStrategy(BaseScoring):
    """
    Evaluates convergent synthesis strategy where two fragments are coupled 
    via a specific reaction type at a target depth.
    """
    
    def __init__(self, config: Dict):
        self.target_depth = config["coupling_step_depth"]
        self.fragment_count = config["fragment_count"]
        self.coupling_reaction = config["coupling_reaction"].lower()
        
        # Define SMARTS patterns for different coupling reactions
        self.coupling_patterns = {
            "suzuki": ["[#6]~[#5]", "[#6]~[Br,I,Cl]"],  # Boronic acid/ester + halide
            "sonogashira": ["[#6]#[#6]", "[#6]~[Br,I,Cl]"],  # Alkyne + halide  
            "heck": ["[#6]=[#6]", "[#6]~[Br,I,Cl]"],  # Alkene + halide
            "ullmann": ["[#7]", "[#6]~[Br,I,Cl]"],  # Amine + halide
            "click": ["[#6]#[#6]", "[#7]~[#7]~[#7]"],  # Alkyne + azide
        }

    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Coupling reaction not found
        
        # Score based on how close the coupling depth is to target
        depth_penalty = abs(x - self.target_depth) * 2
        return max(0, 10 - depth_penalty)

    def hit_condition(self, d) -> bool:
        """Check if this reaction is the target coupling reaction with correct fragment count."""
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            prod_smiles, react_smiles = rxn_smiles.split(">>")
            
            # Parse reactants and product
            product = Chem.MolFromSmiles(prod_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in react_smiles.split(".")]
            
            # Filter out small molecules (catalysts, solvents) - keep only fragments
            fragments = [r for r in reactants if r and r.GetNumAtoms() > 5]
            
            # Check if we have the expected number of fragments
            if len(fragments) != self.fragment_count:
                return False
            
            # Check if this is the specified coupling reaction type
            return self._is_coupling_reaction(fragments, product)
            
        except Exception:
            return False

    def _is_coupling_reaction(self, reactants, product) -> bool:
        """Check if the reaction matches the specified coupling pattern."""
        if self.coupling_reaction not in self.coupling_patterns:
            return False
        
        patterns = self.coupling_patterns[self.coupling_reaction]
        
        # For coupling reactions, we expect complementary functional groups
        # Check if reactants have the required functional groups
        reactant_matches = []
        for pattern in patterns:
            pattern_mol = Chem.MolFromSmarts(pattern)
            matches = [r for r in reactants if r.HasSubstructMatch(pattern_mol)]
            reactant_matches.append(len(matches) > 0)
        
        # For most coupling reactions, we need both functional groups present
        if self.coupling_reaction in ["suzuki", "sonogashira", "heck", "ullmann", "click"]:
            return all(reactant_matches)
        
        return any(reactant_matches)
