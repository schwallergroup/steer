"""Generated evaluation code for: Early convergent fragment coupling via SNAr"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentSNArCoupling(BaseScoring):
    """
    Evaluates synthesis routes for early convergent fragment coupling via nucleophilic aromatic substitution (SNAr).
    Rewards routes where two main building blocks are coupled early through SNAr reactions.
    """
    
    def __init__(self, config: Dict):
        self.timing_threshold = config.get("timing_threshold", 0.3)  # Early means within first 30% of route
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # SNAr coupling doesn't happen
        else:
            # Earlier coupling gets higher score
            # x is depth fraction, so lower x means earlier
            return max(0, 10 * (self.timing_threshold - x) / self.timing_threshold)
    
    def hit_condition(self, d):
        """Check if reaction is a convergent SNAr coupling of two fragments"""
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            rxn_parts = rxn_smiles.split(">>")
            
            if len(rxn_parts) != 2:
                return False
                
            reactants = rxn_parts[0].split(".")
            products = rxn_parts[1].split(".")
            
            # Must be convergent (2 reactants -> 1 product)
            if len(reactants) != 2 or len(products) != 1:
                return False
            
            # Parse molecules
            react_mols = [Chem.MolFromSmiles(r) for r in reactants]
            prod_mol = Chem.MolFromSmiles(products[0])
            
            if None in react_mols or prod_mol is None:
                return False
            
            # Check if it's a nucleophilic aromatic substitution
            return self._is_snar_reaction(react_mols, prod_mol)
            
        except (KeyError, AttributeError, ValueError):
            return False
    
    def _is_snar_reaction(self, reactants, product):
        """Detect if reaction is nucleophilic aromatic substitution"""
        # Look for aromatic electrophile with electron-withdrawing groups
        aromatic_electrophile_patterns = [
            "[c:1][F,Cl,Br,I]",  # Aryl halide
            "[c:1][N+](=O)[O-]",  # Nitro-activated aryl
            "[c:1][C](=O)[OH,OR]",  # Carbonyl-activated aryl
            "[c:1][S](=O)(=O)[OH,OR]"  # Sulfonyl-activated aryl
        ]
        
        # Look for nucleophile patterns
        nucleophile_patterns = [
            "[NH2,NH]",  # Amines
            "[OH]",      # Alcohols/phenols
            "[SH,S-]",   # Thiols/thiolates
            "[C-]"       # Carbanions
        ]
        
        has_electrophile = False
        has_nucleophile = False
        
        for reactant in reactants:
            # Check for aromatic electrophile
            for pattern in aromatic_electrophile_patterns:
                if reactant.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                    has_electrophile = True
                    break
            
            # Check for nucleophile
            for pattern in nucleophile_patterns:
                if reactant.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                    has_nucleophile = True
                    break
        
        # Additional check: product should have aromatic C-Nu bond formation
        # and loss of leaving group
        aromatic_substitution_product = "[c][NH,O,S,C]"
        has_substitution_product = product.HasSubstructMatch(Chem.MolFromSmarts(aromatic_substitution_product))
        
        return has_electrophile and has_nucleophile and has_substitution_product
