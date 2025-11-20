"""Generated evaluation code for: Extensive carboxylic acid derivative cycling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class CarboxylicAcidCycling(MultiRxnCondBase):
    """
    Evaluates routes for extensive carboxylic acid derivative cycling.
    Detects when routes cycle through multiple carboxylic acid derivatives
    (acid -> acyl halide -> ester -> acid -> acyl halide -> amide -> ester -> acid)
    in repetitive patterns.
    """
    
    def __init__(self, config):
        self.min_cycles = config.get("min_cycles", 2)
        
        # SMARTS patterns for carboxylic acid derivatives
        self.carboxyl_patterns = {
            "carboxylic_acid": "[CX3](=[OX1])[OX2H1]",
            "ester": "[CX3](=[OX1])[OX2][CX4]",
            "acyl_chloride": "[CX3](=[OX1])[ClX1]",
            "acyl_fluoride": "[CX3](=[OX1])[FX1]",
            "amide": "[CX3](=[OX1])[NX3]",
            "anhydride": "[CX3](=[OX1])[OX2][CX3](=[OX1])"
        }
        
        # Compile patterns for efficiency
        self.compiled_patterns = {
            name: Chem.MolFromSmarts(pattern) 
            for name, pattern in self.carboxyl_patterns.items()
        }
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        """Check if route shows extensive carboxylic acid derivative cycling."""
        reactions = self.get_rxns(d)
        
        # Track derivative types encountered in sequence
        derivative_sequence = []
        
        for rxn in reactions:
            derivative_type = self.identify_carboxyl_derivative_reaction(rxn)
            if derivative_type:
                derivative_sequence.append(derivative_type)
        
        # Count cycles through different derivatives
        cycle_count = self.count_derivative_cycles(derivative_sequence)
        
        condition = cycle_count >= self.min_cycles
        return condition, len(reactions)
    
    def identify_carboxyl_derivative_reaction(self, rxn_smiles):
        """
        Identify what type of carboxylic acid derivative transformation occurs.
        Returns the predominant derivative type involved.
        """
        try:
            parts = rxn_smiles.split(">>")
            if len(parts) != 2:
                return None
                
            reactants = [Chem.MolFromSmiles(smi) for smi in parts[0].split(".") if smi]
            products = [Chem.MolFromSmiles(smi) for smi in parts[1].split(".") if smi]
            
            if not all(reactants + products):
                return None
            
            # Check for carboxyl derivatives in reactants and products
            reactant_types = set()
            product_types = set()
            
            for mol in reactants:
                for name, pattern in self.compiled_patterns.items():
                    if mol.HasSubstructMatch(pattern):
                        reactant_types.add(name)
            
            for mol in products:
                for name, pattern in self.compiled_patterns.items():
                    if mol.HasSubstructMatch(pattern):
                        product_types.add(name)
            
            # If we have carboxyl derivatives in both reactants and products, 
            # return the most "advanced" derivative type present
            all_types = reactant_types.union(product_types)
            if len(all_types) >= 1:
                # Priority order for derivative complexity
                priority = ["anhydride", "acyl_fluoride", "acyl_chloride", 
                           "amide", "ester", "carboxylic_acid"]
                
                for deriv_type in priority:
                    if deriv_type in all_types:
                        return deriv_type
            
            return None
            
        except Exception:
            return None
    
    def count_derivative_cycles(self, derivative_sequence):
        """
        Count how many times we cycle through different carboxylic acid derivatives.
        A cycle is defined as seeing the same derivative type again after encountering others.
        """
        if len(derivative_sequence) < 3:
            return 0
        
        cycles = 0
        seen_derivatives = set()
        
        for i, derivative in enumerate(derivative_sequence):
            if derivative in seen_derivatives:
                # We've seen this derivative before - potential cycle
                # Check if we've seen other derivatives in between
                last_occurrence = -1
                for j in range(i-1, -1, -1):
                    if derivative_sequence[j] == derivative:
                        last_occurrence = j
                        break
                
                if last_occurrence >= 0:
                    # Check if there are different derivatives between occurrences
                    between_derivatives = set(derivative_sequence[last_occurrence+1:i])
                    if len(between_derivatives) > 0:
                        cycles += 1
            
            seen_derivatives.add(derivative)
        
        return cycles
