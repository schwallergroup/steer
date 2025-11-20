"""Generated evaluation code for: Sequential chemoselective Suzuki couplings"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SequentialChemoselectiveSuzuki(MultiRxnCondBase):
    """
    Evaluates routes for sequential chemoselective Suzuki coupling reactions.
    Checks for multiple Suzuki couplings that exploit differential halide reactivity
    (e.g., I vs Br) for selective bond formation.
    """
    
    def __init__(self, config):
        self.min_count = config.get("min_count", 2)
        self.require_chemoselective = config.get("chemoselective", True)
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        suzuki_reactions = []
        
        # Find all Suzuki coupling reactions
        for i, rxn in enumerate(reactions):
            if self.detect_suzuki_coupling(rxn):
                suzuki_reactions.append((i, rxn))
        
        # Check if we have minimum required count
        if len(suzuki_reactions) < self.min_count:
            return False, len(reactions)
        
        # If chemoselective requirement is enabled, check for differential halides
        if self.require_chemoselective:
            condition = self.check_chemoselectivity(suzuki_reactions)
        else:
            condition = True
            
        return condition, len(reactions)
    
    def detect_suzuki_coupling(self, rxn):
        """Detect Suzuki coupling by looking for boronic acid/ester and halide coupling patterns."""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0].split(".")
        products = rxn_parts[1].split(".")
        
        # Look for boronic acid/ester patterns
        boronic_patterns = [
            "B(O)(O)",  # boronic acid
            "B1OC(C)(C)C(C)(C)O1",  # pinacolborane
            "[B-]([O-])([O-])[O-]"  # borate
        ]
        
        # Look for halide patterns (prefer I > Br > Cl reactivity)
        halide_patterns = ["[I]", "[Br]", "[Cl]"]
        
        has_boronic = False
        has_halide = False
        
        for reactant in reactants:
            try:
                mol = Chem.MolFromSmiles(reactant)
                if mol is None:
                    continue
                    
                # Check for boronic acid/ester
                for pattern in boronic_patterns:
                    pattern_mol = Chem.MolFromSmarts(pattern)
                    if pattern_mol and mol.HasSubstructMatch(pattern_mol):
                        has_boronic = True
                        
                # Check for halide
                for pattern in halide_patterns:
                    pattern_mol = Chem.MolFromSmarts(pattern)
                    if pattern_mol and mol.HasSubstructMatch(pattern_mol):
                        has_halide = True
                        
            except:
                continue
                
        return has_boronic and has_halide
    
    def check_chemoselectivity(self, suzuki_reactions):
        """Check if the sequential Suzuki reactions show chemoselective patterns."""
        if len(suzuki_reactions) < 2:
            return False
            
        # Look for evidence of differential halide reactivity
        halide_hierarchy = {"[I]": 3, "[Br]": 2, "[Cl]": 1, "[F]": 0}
        reaction_halides = []
        
        for depth, rxn in suzuki_reactions:
            rxn_parts = rxn.split(">>")
            reactants = rxn_parts[0].split(".")
            
            max_reactivity = 0
            for reactant in reactants:
                try:
                    mol = Chem.MolFromSmiles(reactant)
                    if mol is None:
                        continue
                        
                    for halide, reactivity in halide_hierarchy.items():
                        pattern_mol = Chem.MolFromSmarts(halide)
                        if pattern_mol and mol.HasSubstructMatch(pattern_mol):
                            max_reactivity = max(max_reactivity, reactivity)
                            
                except:
                    continue
                    
            reaction_halides.append((depth, max_reactivity))
        
        # Sort by reaction depth (earlier reactions first)
        reaction_halides.sort(key=lambda x: x[0])
        
        # Check if there's a pattern of decreasing reactivity or mixed halides
        reactivities = [r[1] for r in reaction_halides]
        
        # Chemoselective if we see different halide reactivities
        unique_reactivities = set(reactivities)
        if len(unique_reactivities) > 1:
            return True
            
        # Also accept if high reactivity halides are used (typical for sequential couplings)
        if any(r >= 2 for r in reactivities):  # Br or I present
            return True
            
        return False
