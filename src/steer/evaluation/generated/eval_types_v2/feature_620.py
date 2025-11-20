"""Generated evaluation code for: Convergent synthesis via two fragments"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentStrategy(MultiRxnCondBase):
    """
    Evaluates convergent synthesis strategy where multiple fragments are built 
    separately and joined via a specific coupling reaction.
    
    Checks if the route contains the specified number of fragments that are
    coupled together using the target coupling reaction type.
    """
    
    def __init__(self, config):
        self.fragment_count = config.get("fragment_count", 2)
        self.coupling_reaction = config.get("coupling_reaction", "suzuki").lower()
        
        # Define SMARTS patterns for different coupling reactions
        self.coupling_patterns = {
            "suzuki": ["[C:1]-[B]", "[C:2]-[Br,I,Cl]"],  # Boronic acid/ester + halide
            "sonogashira": ["[C:1]#[C]", "[C:2]-[Br,I,Cl]"],  # Terminal alkyne + halide
            "heck": ["[C:1]=[C]", "[C:2]-[Br,I,Cl]"],  # Alkene + halide
            "negishi": ["[C:1]-[Zn]", "[C:2]-[Br,I,Cl]"],  # Organozinc + halide
            "stille": ["[C:1]-[Sn]", "[C:2]-[Br,I,Cl]"],  # Organotin + halide
        }
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        """
        Check if the synthesis route follows a convergent strategy with the
        specified coupling reaction and fragment count.
        """
        reactions = self.get_rxns(d)
        total_reactions = len(reactions)
        
        # Look for the coupling reaction
        coupling_found = False
        coupling_depth = -1
        
        for i, rxn in enumerate(reactions):
            if self.detect_coupling_reaction(rxn):
                coupling_found = True
                coupling_depth = i
                break
        
        if not coupling_found:
            return False, total_reactions
        
        # Check if coupling happens with appropriate convergence
        # (should not be too early in the synthesis)
        convergence_threshold = 0.3  # Coupling should happen in later 70% of route
        is_convergent = (coupling_depth / max(1, total_reactions)) >= convergence_threshold
        
        # Verify fragment complexity by checking reactant count at coupling step
        coupling_rxn = reactions[coupling_depth]
        reactant_count = self.count_reactants(coupling_rxn)
        has_sufficient_fragments = reactant_count >= self.fragment_count
        
        condition_met = coupling_found and is_convergent and has_sufficient_fragments
        return condition_met, total_reactions
    
    def detect_coupling_reaction(self, rxn):
        """
        Detect if a reaction is the specified coupling reaction type.
        """
        if self.coupling_reaction not in self.coupling_patterns:
            return False
        
        patterns = self.coupling_patterns[self.coupling_reaction]
        
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
            
            reactants = rxn_parts[0].split(".")
            
            # Check if reactants contain the coupling partners
            pattern_matches = [False] * len(patterns)
            
            for reactant_smiles in reactants:
                mol = Chem.MolFromSmiles(reactant_smiles)
                if mol is None:
                    continue
                
                for i, pattern in enumerate(patterns):
                    pattern_mol = Chem.MolFromSmarts(pattern)
                    if pattern_mol and mol.HasSubstructMatch(pattern_mol):
                        pattern_matches[i] = True
            
            # All patterns should be matched for the coupling reaction
            return all(pattern_matches)
            
        except Exception:
            return False
    
    def count_reactants(self, rxn):
        """
        Count the number of reactants in a reaction.
        """
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return 0
            
            reactants = rxn_parts[0].split(".")
            # Filter out small molecules (catalysts, bases, etc.)
            significant_reactants = []
            
            for reactant_smiles in reactants:
                mol = Chem.MolFromSmiles(reactant_smiles)
                if mol and mol.GetNumAtoms() > 5:  # Filter out small molecules
                    significant_reactants.append(reactant_smiles)
            
            return len(significant_reactants)
            
        except Exception:
            return 0
