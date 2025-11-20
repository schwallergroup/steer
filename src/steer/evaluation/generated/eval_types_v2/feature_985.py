"""Generated evaluation code for: Late stage Williamson ether coupling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageWilliamsonEther(BaseScoring):
    """
    Evaluates whether a Williamson ether synthesis reaction occurs late in the synthetic route.
    Williamson ether synthesis involves nucleophilic substitution between an alkoxide and 
    an alkyl halide/tosylate to form an ether bond.
    """
    
    def __init__(self, config: Dict):
        self.step_position = config.get("step_position", 3)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Williamson ether synthesis doesn't occur
        else:
            # Late-stage reaction is better, so higher depth gives higher score
            # Scale to 0-10 range where later reactions score higher
            return min(10, x * 10)
    
    def hit_condition(self, d) -> bool:
        """
        Detects Williamson ether synthesis by looking for:
        1. Formation of new ether bond (C-O-C)
        2. Breaking of C-X bond where X is halide or tosylate
        3. Presence of alkoxide nucleophile
        """
        metadata = d.get("metadata", {})
        if "mapped_reaction_smiles" not in metadata:
            return False
            
        rxn_smiles = metadata["mapped_reaction_smiles"]
        rxn_parts = rxn_smiles.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        products = rxn_parts[0]
        reactants = rxn_parts[1]
        
        try:
            prod_mol = Chem.MolFromSmiles(products)
            react_mols = [Chem.MolFromSmiles(r) for r in reactants.split(".")]
            
            if not prod_mol or not all(react_mols):
                return False
            
            # Check for ether formation pattern
            if self._has_ether_formation(prod_mol, react_mols):
                return True
                
        except Exception:
            return False
            
        return False
    
    def _has_ether_formation(self, product, reactants) -> bool:
        """
        Check if the reaction represents Williamson ether synthesis
        """
        # Look for ether pattern in product
        ether_pattern = Chem.MolFromSmarts("[C:1]-[O:2]-[C:3]")
        if not product.HasSubstructMatch(ether_pattern):
            return False
        
        # Check reactants for typical Williamson ether precursors
        has_alkyl_halide = False
        has_alkoxide = False
        
        # Patterns for alkyl halides/tosylates
        halide_patterns = [
            "[C:1]-[Cl,Br,I]",  # Alkyl halides
            "[C:1]-[O:2]-[S:3](=[O:4])(=[O:5])-[c:6]1[c:7][c:8][c:9]([CH3:10])[c:11][c:12]1"  # Tosylates
        ]
        
        # Patterns for alkoxides (often as metal alkoxides)
        alkoxide_patterns = [
            "[C:1]-[O-:2]",  # Alkoxide anion
            "[C:1]-[O:2]-[Na,K,Li,Cs]",  # Metal alkoxide
            "[C:1]-[OH:2]"  # Alcohol (with base present)
        ]
        
        for reactant in reactants:
            # Check for alkyl halide/tosylate
            for pattern_smarts in halide_patterns:
                pattern = Chem.MolFromSmarts(pattern_smarts)
                if pattern and reactant.HasSubstructMatch(pattern):
                    has_alkyl_halide = True
                    break
            
            # Check for alkoxide
            for pattern_smarts in alkoxide_patterns:
                pattern = Chem.MolFromSmarts(pattern_smarts)
                if pattern and reactant.HasSubstructMatch(pattern):
                    has_alkoxide = True
                    break
        
        # Also check for presence of base (common in Williamson ether synthesis)
        base_patterns = [
            "[Na,K,Li,Cs][OH]",  # Metal hydroxides
            "[N:1]([C:2])([C:3])[C:4]",  # Tertiary amines
            "c1[c:1][c:2][c:3]([N:4]([C:5])[C:6])[c:7][c:8]1"  # Aromatic amines
        ]
        
        has_base = False
        for reactant in reactants:
            for pattern_smarts in base_patterns:
                pattern = Chem.MolFromSmarts(pattern_smarts)
                if pattern and reactant.HasSubstructMatch(pattern):
                    has_base = True
                    break
        
        # Williamson ether synthesis requires alkyl halide and nucleophilic oxygen
        # Accept if we have alcohol + base as equivalent to alkoxide
        return has_alkyl_halide and (has_alkoxide or has_base)
