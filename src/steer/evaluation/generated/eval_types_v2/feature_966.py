"""Generated evaluation code for: Early azide introduction via mesylate displacement"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlyAzideMesylateDisplacement(BaseScoring):
    """
    Evaluates routes for early introduction of azide functionality via nucleophilic 
    substitution of mesylate leaving groups. Returns higher scores when the 
    azide-mesylate displacement occurs earlier in the synthesis.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "fractional")
        self.target_depth = config.get("target_depth", {}).get("value", 0.3)  # Early stage default
        
    def route_scoring(self, x) -> float:
        """
        Convert depth fraction to score (0-10 scale).
        Earlier reactions get higher scores.
        """
        if x < 0:
            return 0  # Reaction doesn't occur
        else:
            # Early stage reactions preferred - invert the depth fraction
            return 1 - x
            
    def hit_condition(self, d) -> bool:
        """
        Check if a reaction node represents azide displacement of mesylate.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            prod_smiles, react_smiles = mapped_rxn.split(">>")
            product = Chem.MolFromSmiles(prod_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in react_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
                
            # Check for azide introduction (N=[N+]=[N-] or [N-][N+]#N)
            azide_patterns = [
                Chem.MolFromSmarts("[N-][N+]#[N]"),  # Azide anion form 1
                Chem.MolFromSmarts("[N]=[N+]=[N-]"),  # Azide anion form 2
                Chem.MolFromSmarts("[C][N]=[N+]=[N-]"),  # Organic azide
                Chem.MolFromSmarts("[C][N+]#[N][N-]")   # Alternative organic azide
            ]
            
            # Check if product contains azide
            has_azide_product = any(product.HasSubstructMatch(pattern) for pattern in azide_patterns if pattern)
            
            if not has_azide_product:
                return False
                
            # Check for mesylate leaving group in reactants
            # Mesylate: R-O-S(=O)(=O)-CH3
            mesylate_pattern = Chem.MolFromSmarts("[C,c][O][S](=[O])(=[O])[C]")
            has_mesylate_reactant = any(r.HasSubstructMatch(mesylate_pattern) for r in reactants if mesylate_pattern)
            
            if not has_mesylate_reactant:
                return False
                
            # Check for azide source in reactants (typically NaN3 or similar)
            azide_source_patterns = [
                Chem.MolFromSmarts("[Na+].[N-][N+]#[N]"),  # Sodium azide
                Chem.MolFromSmarts("[N-][N+]#[N]"),        # Azide anion
                Chem.MolFromSmarts("[K+].[N-][N+]#[N]")    # Potassium azide
            ]
            
            # Check if any reactant is an azide source or if azide anion is present
            has_azide_source = False
            for reactant in reactants:
                if any(reactant.HasSubstructMatch(pattern) for pattern in azide_source_patterns if pattern):
                    has_azide_source = True
                    break
                # Also check for simple azide anion patterns
                for azide_pattern in azide_patterns:
                    if azide_pattern and reactant.HasSubstructMatch(azide_pattern):
                        has_azide_source = True
                        break
                        
            return has_mesylate_reactant and has_azide_source
            
        except Exception:
            return False
