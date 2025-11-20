"""Generated evaluation code for: Multiple ester group interconversion cycles"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EsterInterconversionCycles(MultiRxnCondBase):
    """
    Detects multiple ester group interconversion cycles in synthesis routes.
    
    Identifies redundant ester-to-acid-to-ester transformations that unnecessarily
    change ester groups through hydrolysis/esterification cycles.
    """
    
    def __init__(self, config):
        self.min_cycles = config.get("min_cycles", 2)
        self.ester_pattern = Chem.MolFromSmarts("[C](=O)[O][C]")  # Ester functional group
        self.carboxylic_acid_pattern = Chem.MolFromSmarts("[C](=O)[O][H]")  # Carboxylic acid
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        cycles = self.count_ester_cycles(reactions)
        
        condition = cycles >= self.min_cycles
        return condition, len(reactions)
    
    def count_ester_cycles(self, reactions) -> int:
        """Count the number of ester interconversion cycles in the reaction sequence."""
        ester_transformations = []
        
        for rxn in reactions:
            transformation = self.classify_ester_transformation(rxn)
            if transformation:
                ester_transformations.append(transformation)
        
        # Count cycles by looking for alternating hydrolysis/esterification patterns
        cycles = 0
        i = 0
        while i < len(ester_transformations) - 1:
            current = ester_transformations[i]
            
            # Look for a cycle starting with hydrolysis
            if current == "hydrolysis":
                # Find the next esterification that could complete the cycle
                for j in range(i + 1, len(ester_transformations)):
                    if ester_transformations[j] == "esterification":
                        # Check if this creates a redundant cycle by comparing ester groups
                        if self.is_redundant_cycle(reactions, i, j):
                            cycles += 1
                            i = j  # Move past this cycle
                            break
                else:
                    i += 1
            else:
                i += 1
        
        return cycles
    
    def classify_ester_transformation(self, rxn) -> str:
        """Classify reaction as ester hydrolysis, esterification, or neither."""
        try:
            rxn_parts = rxn.split(">>")
            reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[0].split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in rxn_parts[1].split(".")]
            
            reactant_esters = sum(1 for mol in reactants if mol and mol.HasSubstructMatch(self.ester_pattern))
            reactant_acids = sum(1 for mol in reactants if mol and mol.HasSubstructMatch(self.carboxylic_acid_pattern))
            
            product_esters = sum(1 for mol in products if mol and mol.HasSubstructMatch(self.ester_pattern))
            product_acids = sum(1 for mol in products if mol and mol.HasSubstructMatch(self.carboxylic_acid_pattern))
            
            # Hydrolysis: ester -> carboxylic acid
            if reactant_esters > product_esters and product_acids > reactant_acids:
                return "hydrolysis"
            
            # Esterification: carboxylic acid -> ester
            if reactant_acids > product_acids and product_esters > reactant_esters:
                return "esterification"
            
            return None
            
        except Exception:
            return None
    
    def is_redundant_cycle(self, reactions, hydrolysis_idx, esterification_idx) -> bool:
        """
        Check if hydrolysis followed by esterification creates a redundant cycle
        by comparing the ester groups before and after the cycle.
        """
        try:
            # Get the molecule before hydrolysis
            hydrolysis_rxn = reactions[hydrolysis_idx].split(">>")
            pre_hydrolysis = Chem.MolFromSmiles(hydrolysis_rxn[0].split(".")[0])
            
            # Get the molecule after esterification
            esterification_rxn = reactions[esterification_idx].split(">>")
            post_esterification = Chem.MolFromSmiles(esterification_rxn[1].split(".")[0])
            
            if not pre_hydrolysis or not post_esterification:
                return False
            
            # Compare ester substitution patterns
            pre_ester_atoms = self.get_ester_substitution_pattern(pre_hydrolysis)
            post_ester_atoms = self.get_ester_substitution_pattern(post_esterification)
            
            # If ester patterns are different, it's a meaningful transformation
            # If they're the same or very similar, it's likely redundant
            return self.compare_ester_patterns(pre_ester_atoms, post_ester_atoms)
            
        except Exception:
            return False
    
    def get_ester_substitution_pattern(self, mol):
        """Extract the substitution pattern around ester groups."""
        ester_patterns = []
        matches = mol.GetSubstructMatches(self.ester_pattern)
        
        for match in matches:
            # Get the carbon attached to the ester oxygen (R group)
            ester_oxygen_idx = match[2]  # Oxygen in [C](=O)[O][C]
            ester_carbon_idx = match[3]   # Carbon in [C](=O)[O][C]
            
            ester_carbon = mol.GetAtomWithIdx(ester_carbon_idx)
            # Get neighbor pattern (simplified)
            neighbors = [n.GetSymbol() for n in ester_carbon.GetNeighbors() if n.GetIdx() != ester_oxygen_idx]
            ester_patterns.append(tuple(sorted(neighbors)))
        
        return ester_patterns
    
    def compare_ester_patterns(self, pattern1, pattern2) -> bool:
        """Compare ester patterns to determine if cycle is redundant."""
        if len(pattern1) != len(pattern2):
            return False
        
        # Simple comparison - in practice, this could be more sophisticated
        pattern1_sorted = sorted(pattern1)
        pattern2_sorted = sorted(pattern2)
        
        # If patterns are very similar, consider it redundant
        similarity = sum(1 for p1, p2 in zip(pattern1_sorted, pattern2_sorted) if p1 == p2)
        return similarity >= len(pattern1) * 0.8  # 80% similarity threshold
