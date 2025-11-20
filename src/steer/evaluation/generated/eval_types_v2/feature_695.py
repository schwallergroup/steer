"""Generated evaluation code for: Late stage N-alkylation of lactam"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageNAlkylationLactam(BaseScoring):
    """
    Evaluates whether N-alkylation of a lactam occurs at a late stage in the synthesis route.
    
    Checks for N-alkylation reactions involving lactam substrates (containing N-C=O pattern)
    and rewards routes where this transformation happens late in the synthesis.
    """
    
    def __init__(self, config: Dict):
        self.substrate_pattern = config["parameters"]["substrate_pattern"]
        self.depth_threshold = config["parameters"]["depth_threshold"]
        self.lactam_mol = Chem.MolFromSmarts(self.substrate_pattern)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # N-alkylation doesn't happen
        else:
            # Late-stage is better, penalize if occurs before depth threshold
            if x <= self.depth_threshold / 10.0:  # Convert to fraction
                return 10 * (1 - x)  # Higher score for later occurrence
            else:
                return 5 * (1 - x)  # Lower reward for very early occurrence
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction node represents N-alkylation of a lactam.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles")
        
        if not mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            product = Chem.MolFromSmiles(rxn_parts[0])
            reactants = [Chem.MolFromSmiles(r) for r in rxn_parts[1].split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check if product contains lactam pattern
            if not product.HasSubstructMatch(self.lactam_mol):
                return False
            
            # Check if any reactant contains lactam pattern (substrate)
            lactam_reactant = None
            alkylating_agent = None
            
            for reactant in reactants:
                if reactant.HasSubstructMatch(self.lactam_mol):
                    lactam_reactant = reactant
                else:
                    # Potential alkylating agent
                    alkylating_agent = reactant
            
            if not lactam_reactant or not alkylating_agent:
                return False
            
            # Check for N-alkylation: new C-N bond formation
            return self._is_n_alkylation(product, lactam_reactant, alkylating_agent)
            
        except Exception:
            return False
    
    def _is_n_alkylation(self, product, lactam_reactant, alkylating_agent) -> bool:
        """
        Verify that the reaction represents N-alkylation by checking for new C-N bond formation.
        """
        try:
            # Find nitrogen atoms in lactam pattern in both reactant and product
            lactam_matches_reactant = lactam_reactant.GetSubstructMatches(self.lactam_mol)
            lactam_matches_product = product.GetSubstructMatches(self.lactam_mol)
            
            if not lactam_matches_reactant or not lactam_matches_product:
                return False
            
            # Get the nitrogen atom index from the pattern match
            # In pattern [#7]C(=O), nitrogen is at index 0
            for match_r in lactam_matches_reactant:
                n_idx_reactant = match_r[0]
                n_atom_reactant = lactam_reactant.GetAtomWithIdx(n_idx_reactant)
                
                for match_p in lactam_matches_product:
                    n_idx_product = match_p[0]
                    n_atom_product = product.GetAtomWithIdx(n_idx_product)
                    
                    # Compare atom map numbers to ensure we're looking at the same nitrogen
                    if (n_atom_reactant.GetAtomMapNum() > 0 and 
                        n_atom_reactant.GetAtomMapNum() == n_atom_product.GetAtomMapNum()):
                        
                        # Count carbon neighbors of nitrogen
                        c_neighbors_reactant = sum(1 for neighbor in n_atom_reactant.GetNeighbors() 
                                                 if neighbor.GetSymbol() == 'C')
                        c_neighbors_product = sum(1 for neighbor in n_atom_product.GetNeighbors() 
                                                if neighbor.GetSymbol() == 'C')
                        
                        # N-alkylation should increase carbon neighbors by 1
                        if c_neighbors_product == c_neighbors_reactant + 1:
                            return True
            
            return False
            
        except Exception:
            return False
