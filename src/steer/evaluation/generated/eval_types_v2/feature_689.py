"""Generated evaluation code for: Convergent synthesis via two fragments"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentSynthesis(BaseScoring):
    """
    Evaluates convergent synthesis strategies where two major fragments are combined
    via a specified coupling reaction type.
    """
    
    def __init__(self, config: Dict):
        self.fragment_count = config["parameters"]["fragment_count"]
        self.coupling_reaction = config["parameters"]["coupling_reaction"]
        
        # Define SMARTS patterns for nucleophilic aromatic substitution
        self.snar_patterns = {
            "electrophile": "[c:1][F,Cl,Br,I]",  # Aryl halide
            "nucleophile": "[N,O,S:2][H]",       # Nucleophile with hydrogen
            "product": "[c:1][N,O,S:2]"          # Coupled product
        }
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Convergent reaction doesn't happen
        else:
            # Earlier convergent coupling is better (more convergent strategy)
            return (1 - x) * 10
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction represents a convergent coupling."""
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            prod_smiles, react_smiles = rxn_smiles.split(">>")
            
            # Parse molecules
            product = Chem.MolFromSmiles(prod_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in react_smiles.split(".")]
            reactants = [r for r in reactants if r is not None]
            
            if not product or len(reactants) != self.fragment_count:
                return False
            
            # Check if this is the specified coupling reaction type
            if self.coupling_reaction == "nucleophilic_aromatic_substitution":
                return self._is_snar_reaction(product, reactants)
            
            # For other coupling types, check basic convergent criteria
            return self._is_convergent_coupling(product, reactants)
            
        except Exception:
            return False
    
    def _is_snar_reaction(self, product, reactants) -> bool:
        """Check if reaction matches SNAr pattern."""
        if len(reactants) != 2:
            return False
        
        # Look for electrophile and nucleophile patterns in reactants
        electrophile_pattern = Chem.MolFromSmarts(self.snar_patterns["electrophile"])
        nucleophile_pattern = Chem.MolFromSmarts(self.snar_patterns["nucleophile"])
        product_pattern = Chem.MolFromSmarts(self.snar_patterns["product"])
        
        if not all([electrophile_pattern, nucleophile_pattern, product_pattern]):
            return False
        
        # Check if one reactant has electrophile and other has nucleophile
        has_electrophile = [r.HasSubstructMatch(electrophile_pattern) for r in reactants]
        has_nucleophile = [r.HasSubstructMatch(nucleophile_pattern) for r in reactants]
        
        # Exactly one reactant should have electrophile, one should have nucleophile
        if sum(has_electrophile) == 1 and sum(has_nucleophile) == 1:
            # Check if product has the expected coupling pattern
            return product.HasSubstructMatch(product_pattern)
        
        return False
    
    def _is_convergent_coupling(self, product, reactants) -> bool:
        """General check for convergent coupling based on fragment size."""
        if len(reactants) != self.fragment_count:
            return False
        
        # Check that both fragments are substantial (at least 5 heavy atoms each)
        min_fragment_size = 5
        fragment_sizes = [r.GetNumHeavyAtoms() for r in reactants]
        
        if all(size >= min_fragment_size for size in fragment_sizes):
            # Check that combined fragment size is close to product size
            total_fragment_atoms = sum(fragment_sizes)
            product_atoms = product.GetNumHeavyAtoms()
            
            # Allow for loss of 1-2 atoms (leaving groups, etc.)
            return abs(total_fragment_atoms - product_atoms) <= 2
        
        return False
