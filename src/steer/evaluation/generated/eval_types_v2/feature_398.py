"""Generated evaluation code for: Convergent synthesis via two major fragments"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentSynthesis(MultiRxnCondBase):
    """
    Evaluates convergent synthesis strategies by detecting coupling reactions
    that join major fragments at specified convergence points in the route.
    """
    
    def __init__(self, config):
        self.fragment_count = config.get("fragment_count", 2)
        self.coupling_reactions = config.get("coupling_reactions", [])
        self.convergence_point = config.get("convergence_point", "early")  # early, middle, late
        
        # Define reaction patterns for detection
        self.reaction_patterns = {
            "suzuki": {
                "boronic_acid": "[#6]-B(-O)-O",
                "halide": "[#6]-[Cl,Br,I]",
                "product": "[#6]-[#6]"  # C-C bond formation
            },
            "snar": {
                "nucleophile": "[N,O,S][#6]",
                "electrophile": "[#6]([F,Cl,Br,I])[#6]=[#6]",  # activated aryl halide
                "product": "[N,O,S]-[#6]"
            }
        }
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        """Check if convergent synthesis occurs at the target convergence point"""
        reactions = self.get_rxns(d)
        total_reactions = len(reactions)
        
        # Find coupling reactions and their positions
        coupling_positions = []
        for i, rxn in enumerate(reactions):
            if self.is_coupling_reaction(rxn):
                coupling_positions.append(i)
        
        # Check if we have the right number of coupling reactions
        if len(coupling_positions) < 1:
            return False, total_reactions
        
        # Determine convergence point based on position in route
        convergence_depth = self.get_convergence_depth(coupling_positions, total_reactions)
        target_met = self.meets_convergence_criteria(convergence_depth)
        
        return target_met, total_reactions
    
    def is_coupling_reaction(self, rxn_smiles):
        """Detect if reaction is one of the specified coupling reactions"""
        for reaction_type in self.coupling_reactions:
            if reaction_type in self.reaction_patterns:
                if self.detect_coupling_type(rxn_smiles, reaction_type):
                    return True
        return False
    
    def detect_coupling_type(self, rxn_smiles, reaction_type):
        """Detect specific coupling reaction type"""
        try:
            reactants_smiles, product_smiles = rxn_smiles.split(">>")
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            product = Chem.MolFromSmiles(product_smiles)
            
            if not all(reactants) or not product:
                return False
            
            patterns = self.reaction_patterns[reaction_type]
            
            if reaction_type == "suzuki":
                return self.detect_suzuki(reactants, product, patterns)
            elif reaction_type == "snar":
                return self.detect_snar(reactants, product, patterns)
                
        except Exception:
            return False
        
        return False
    
    def detect_suzuki(self, reactants, product, patterns):
        """Detect Suzuki coupling (boronic acid + halide -> C-C bond)"""
        boronic_pattern = Chem.MolFromSmarts(patterns["boronic_acid"])
        halide_pattern = Chem.MolFromSmarts(patterns["halide"])
        
        has_boronic = any(r.HasSubstructMatch(boronic_pattern) for r in reactants)
        has_halide = any(r.HasSubstructMatch(halide_pattern) for r in reactants)
        
        # Check if product has more C-C bonds than individual reactants
        if has_boronic and has_halide:
            product_atoms = product.GetNumAtoms()
            total_reactant_atoms = sum(r.GetNumAtoms() for r in reactants)
            # Simple heuristic: product should have similar atom count (accounting for lost B, halide)
            return abs(product_atoms - (total_reactant_atoms - 3)) <= 2
        
        return False
    
    def detect_snar(self, reactants, product, patterns):
        """Detect nucleophilic aromatic substitution"""
        nucleophile_pattern = Chem.MolFromSmarts(patterns["nucleophile"])
        electrophile_pattern = Chem.MolFromSmarts(patterns["electrophile"])
        product_pattern = Chem.MolFromSmarts(patterns["product"])
        
        has_nucleophile = any(r.HasSubstructMatch(nucleophile_pattern) for r in reactants)
        has_electrophile = any(r.HasSubstructMatch(electrophile_pattern) for r in reactants)
        has_product_bond = product.HasSubstructMatch(product_pattern)
        
        return has_nucleophile and has_electrophile and has_product_bond
    
    def get_convergence_depth(self, coupling_positions, total_reactions):
        """Calculate relative depth of convergence in the route"""
        if not coupling_positions:
            return -1
        
        # Use the earliest coupling reaction as convergence point
        earliest_coupling = min(coupling_positions)
        return earliest_coupling / max(total_reactions - 1, 1)
    
    def meets_convergence_criteria(self, convergence_depth):
        """Check if convergence occurs at the target point"""
        if convergence_depth < 0:
            return False
        
        if self.convergence_point == "early":
            return convergence_depth <= 0.33
        elif self.convergence_point == "middle":
            return 0.33 < convergence_depth <= 0.67
        elif self.convergence_point == "late":
            return convergence_depth > 0.67
        
        return True
    
    def route_scoring(self, x):
        """Score the route based on convergent synthesis success"""
        if x < 0:
            return 0  # No convergent coupling found
        else:
            return 8 + 2 * (1 - x)  # Higher score for successful convergence
