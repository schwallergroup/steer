"""Generated evaluation code for: Convergent synthesis via fragment coupling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentStrategy(BaseScoring):
    """
    Evaluates convergent synthesis strategies by detecting fragment coupling reactions
    at specific timing in the synthesis route.
    """
    
    def __init__(self, config: Dict):
        self.coupling_reaction = config["parameters"]["coupling_reaction"]
        self.fragment_count = config["parameters"]["fragment_count"]
        self.timing = config["parameters"]["timing"]
        
        # Define SMARTS patterns for different coupling reactions
        self.coupling_patterns = {
            "esterification": "[C:1](=[O:2])[O:3][C:4]",
            "amidation": "[C:1](=[O:2])[N:3]",
            "suzuki": "[c:1][c:2]",  # Aromatic C-C coupling
            "click": "[C:1]1[N:2][N:3][N:4][C:5]1",  # Triazole formation
            "olefin_metathesis": "[C:1]=[C:2]"
        }
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Coupling reaction not found
        
        # Score based on timing preference
        if self.timing == "early":
            return 1 - x  # Earlier is better
        elif self.timing == "mid":
            # Prefer middle timing (around 0.5 depth)
            return 1 - abs(x - 0.5) * 2
        elif self.timing == "late":
            return x  # Later is better
        else:
            return 1 - x  # Default to early preference
    
    def hit_condition(self, d):
        """Check if this reaction represents the target coupling reaction"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
        
        rxn_parts = mapped_rxn.split(">>")
        product_smiles = rxn_parts[0]
        reactants_smiles = rxn_parts[1]
        
        try:
            product_mol = Chem.MolFromSmiles(product_smiles)
            reactant_mols = [Chem.MolFromSmiles(r.strip()) 
                           for r in reactants_smiles.split(".")]
            
            if not product_mol or len(reactant_mols) < self.fragment_count:
                return False
            
            # Check if we have the expected number of substantial fragments
            substantial_fragments = [mol for mol in reactant_mols 
                                   if mol and mol.GetNumHeavyAtoms() >= 5]
            
            if len(substantial_fragments) < self.fragment_count:
                return False
            
            # Check for the specific coupling reaction pattern
            coupling_pattern = self.coupling_patterns.get(self.coupling_reaction)
            if not coupling_pattern:
                return False
            
            pattern_mol = Chem.MolFromSmarts(coupling_pattern)
            if not pattern_mol:
                return False
            
            # Check if product contains the coupling pattern
            if not product_mol.HasSubstructMatch(pattern_mol):
                return False
            
            # Verify the coupling pattern is newly formed (not in reactants)
            pattern_in_reactants = any(
                reactant.HasSubstructMatch(pattern_mol) 
                for reactant in substantial_fragments
            )
            
            return not pattern_in_reactants
            
        except Exception:
            return False
