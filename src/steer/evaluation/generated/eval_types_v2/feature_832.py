"""Generated evaluation code for: Early stage N-alkylation of heterocycle"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlyNAlkylationHeterocycle(BaseScoring):
    """
    Evaluates whether N-alkylation of imidazopyrazine heterocycle occurs early in the synthesis route.
    Returns higher scores when the N-alkylation reaction happens at early stages (lower depth).
    """
    
    def __init__(self, config: Dict):
        self.timing = config.get("timing", "early")
        self.substrate_pattern = "n1cnc2nccnc12"  # imidazopyrazine core SMARTS
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # N-alkylation doesn't happen
        else:
            if self.timing == "early":
                return 1 - x  # Early stage (low depth) is better
            else:
                return x  # Late stage (high depth) is better
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction is N-alkylation of imidazopyrazine"""
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
                
            reactants_smiles, product_smiles = rxn_smiles.split(">>")
            
            # Parse molecules
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
                
            # Check if product contains imidazopyrazine core
            imidazopyrazine_pattern = Chem.MolFromSmarts(self.substrate_pattern)
            if not product.HasSubstructMatch(imidazopyrazine_pattern):
                return False
                
            # Check if any reactant contains imidazopyrazine core
            has_heterocycle_reactant = False
            for reactant in reactants:
                if reactant.HasSubstructMatch(imidazopyrazine_pattern):
                    has_heterocycle_reactant = True
                    break
                    
            if not has_heterocycle_reactant:
                return False
                
            # Check for N-alkylation pattern: nitrogen gains alkyl group
            # Look for nitrogen atoms in imidazopyrazine that gained substituents
            product_matches = product.GetSubstructMatches(imidazopyrazine_pattern)
            
            for reactant in reactants:
                if reactant.HasSubstructMatch(imidazopyrazine_pattern):
                    reactant_matches = reactant.GetSubstructMatches(imidazopyrazine_pattern)
                    
                    # Compare nitrogen substitution patterns
                    for prod_match in product_matches:
                        for react_match in reactant_matches:
                            # Check if nitrogen atoms (positions 0, 2, 4, 7 in imidazopyrazine) gained alkyl groups
                            n_positions = [0, 2, 4, 7]  # Nitrogen positions in the SMARTS pattern
                            
                            for n_pos in n_positions:
                                if n_pos < len(prod_match) and n_pos < len(react_match):
                                    prod_n_atom = product.GetAtomWithIdx(prod_match[n_pos])
                                    react_n_atom = reactant.GetAtomWithIdx(react_match[n_pos])
                                    
                                    # Check if nitrogen gained carbon neighbors (alkylation)
                                    prod_c_neighbors = sum(1 for neighbor in prod_n_atom.GetNeighbors() 
                                                         if neighbor.GetSymbol() == 'C')
                                    react_c_neighbors = sum(1 for neighbor in react_n_atom.GetNeighbors() 
                                                          if neighbor.GetSymbol() == 'C')
                                    
                                    if prod_c_neighbors > react_c_neighbors:
                                        return True
                                        
            return False
            
        except Exception:
            return False
