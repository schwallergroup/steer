"""Generated evaluation code for: Late stage palladium-catalyzed C-N coupling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStagePdCNCoupling(BaseScoring):
    """
    Evaluates if a Buchwald-Hartwig palladium-catalyzed C-N coupling occurs at a late stage.
    Rewards routes where this coupling happens in the final steps to join complex fragments.
    """
    
    def __init__(self, config: Dict):
        self.timing = config.get("timing", "late")
        self.step_position = config.get("step_position", "final")
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Reaction doesn't happen
        
        if self.timing == "late" and self.step_position == "final":
            # Reward very late stage coupling (closer to 0 is better)
            if x <= 0.2:  # Within first 20% of route (very late)
                return 1.0
            elif x <= 0.4:  # Within first 40% of route
                return 0.7
            else:
                return 0.3  # Earlier stage coupling gets lower score
        else:
            # For other timing preferences, use inverse relationship
            return 1 - x
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction is a Buchwald-Hartwig C-N coupling"""
        metadata = d.get("metadata", {})
        rxn_smiles = metadata.get("mapped_reaction_smiles", "")
        
        if not rxn_smiles or ">>" not in rxn_smiles:
            return False
            
        try:
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            if not all(reactants) or not all(products):
                return False
                
            # Check for key patterns of Buchwald-Hartwig coupling
            return self._is_buchwald_hartwig_coupling(reactants, products)
            
        except Exception:
            return False
    
    def _is_buchwald_hartwig_coupling(self, reactants, products) -> bool:
        """Detect Buchwald-Hartwig C-N coupling pattern"""
        
        # Common aryl halide patterns (Ar-X where X = Cl, Br, I)
        aryl_halide_patterns = [
            "[cH0:1][Cl,Br,I:2]",  # Aromatic carbon with halogen
            "[c:1][Cl,Br,I:2]"     # Alternative aromatic pattern
        ]
        
        # Amine nucleophile patterns
        amine_patterns = [
            "[NH2:3][c,C]",        # Primary aromatic/aliphatic amine
            "[NH1:3]([c,C])[c,C]", # Secondary amine
            "[NH2:3]",             # Simple primary amine
            "[NH1:3][c,C]"         # Simple secondary amine
        ]
        
        # C-N bond formation pattern in product
        cn_bond_pattern = "[c:1][NH1,NH0:3]"
        
        # Check if we have aryl halide and amine in reactants
        has_aryl_halide = False
        has_amine = False
        
        for reactant in reactants:
            if not reactant:
                continue
                
            # Check for aryl halide
            for pattern in aryl_halide_patterns:
                if reactant.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                    has_aryl_halide = True
                    break
            
            # Check for amine
            for pattern in amine_patterns:
                if reactant.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                    has_amine = True
                    break
        
        # Check for C-N bond in products
        has_cn_bond = False
        for product in products:
            if product and product.HasSubstructMatch(Chem.MolFromSmarts(cn_bond_pattern)):
                has_cn_bond = True
                break
        
        # Must have all three components for Buchwald-Hartwig coupling
        return has_aryl_halide and has_amine and has_cn_bond
