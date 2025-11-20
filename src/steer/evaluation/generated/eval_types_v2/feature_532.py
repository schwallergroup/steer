"""Generated evaluation code for: Late stage Buchwald-Hartwig amination coupling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageBuchwaldHartwig(BaseScoring):
    """
    Evaluates whether a Buchwald-Hartwig amination coupling occurs in the late stages
    of a synthesis route (within the specified depth threshold).
    
    Buchwald-Hartwig reactions form C-N bonds between aryl halides/triflates and amines
    using palladium catalysis, enabling convergent synthetic strategies.
    """
    
    def __init__(self, config: Dict):
        self.depth_threshold = config.get("depth_threshold", 2)
        
        # SMARTS patterns for Buchwald-Hartwig reaction recognition
        # Aryl halide/triflate + amine -> aryl amine
        self.aryl_halide_patterns = [
            "[cH0,c:1][Cl,Br,I]",  # Aryl chloride, bromide, iodide
            "[cH0,c:1][S](=[O])(=[O])[CF3]"  # Aryl triflate
        ]
        
        self.amine_patterns = [
            "[NH2:2]",  # Primary amine
            "[NH1:2]([CH3,c])",  # Secondary amine
            "[n:2]1[cH][cH][cH][cH][cH]1"  # Aniline-type
        ]
        
        self.product_pattern = "[cH0,c:1][NH0,NH1:2]"  # Aryl-nitrogen bond

    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Reaction doesn't occur
        
        # Convert depth fraction to score favoring late-stage reactions
        if x <= self.depth_threshold / 10.0:  # Within depth threshold
            return 10 * (1 - x)  # Later is better
        else:
            return 5 * (1 - x)  # Still reward occurrence but less

    def hit_condition(self, d) -> bool:
        """Check if this reaction node represents a Buchwald-Hartwig amination."""
        metadata = d.get("metadata", {})
        rxn_smiles = metadata.get("mapped_reaction_smiles")
        
        if not rxn_smiles:
            return False
            
        try:
            rxn_parts = rxn_smiles.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            product_smiles = rxn_parts[0]
            reactants_smiles = rxn_parts[1].split(".")
            
            product_mol = Chem.MolFromSmiles(product_smiles)
            reactant_mols = [Chem.MolFromSmiles(r) for r in reactants_smiles]
            
            if not product_mol or not all(reactant_mols):
                return False
            
            # Check if product contains aryl-amine bond
            product_pattern = Chem.MolFromSmarts(self.product_pattern)
            if not product_mol.HasSubstructMatch(product_pattern):
                return False
            
            # Check if reactants contain aryl halide/triflate and amine
            has_aryl_halide = False
            has_amine = False
            
            for reactant in reactant_mols:
                # Check for aryl halide/triflate
                for pattern_smarts in self.aryl_halide_patterns:
                    pattern = Chem.MolFromSmarts(pattern_smarts)
                    if pattern and reactant.HasSubstructMatch(pattern):
                        has_aryl_halide = True
                        break
                
                # Check for amine
                for pattern_smarts in self.amine_patterns:
                    pattern = Chem.MolFromSmarts(pattern_smarts)
                    if pattern and reactant.HasSubstructMatch(pattern):
                        has_amine = True
                        break
            
            return has_aryl_halide and has_amine
            
        except Exception:
            return False
