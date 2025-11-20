"""Generated evaluation code for: Late sulfonamide formation via activated ester"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateSulfonamideFormation(BaseScoring):
    """
    Evaluates whether sulfonamide formation occurs late in the synthesis route,
    specifically looking for sulfonamide bond formation via activated ester
    (pentafluorophenyl sulfonate) in the final step.
    """
    
    def __init__(self, config: Dict):
        self.step_position_from_end = config.get("step_position_from_end", 1)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Sulfonamide formation doesn't happen
        else:
            # Late-stage sulfonamide formation is better
            # x represents depth fraction, so 1-x gives higher score for later steps
            return 1 - x
            
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction step involves sulfonamide formation via activated ester
        """
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            rxn_parts = rxn_smiles.split(">>")
            
            if len(rxn_parts) != 2:
                return False
                
            reactants_smiles = rxn_parts[0]
            products_smiles = rxn_parts[1]
            
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            
            if not all(reactants) or not all(products):
                return False
            
            # Check for sulfonamide formation pattern
            return self._is_sulfonamide_formation(reactants, products)
            
        except Exception:
            return False
    
    def _is_sulfonamide_formation(self, reactants, products) -> bool:
        """
        Detect sulfonamide formation by checking for:
        1. Activated sulfonate ester in reactants (pentafluorophenyl group)
        2. Amine in reactants
        3. Sulfonamide bond in products
        4. Loss of pentafluorophenol
        """
        # Patterns for detection
        activated_sulfonate_pattern = Chem.MolFromSmarts("[S](=O)(=O)[O][c]1[c]([F])[c]([F])[c]([F])[c]([F])[c]1[F]")
        amine_pattern = Chem.MolFromSmarts("[N;H2,H1;!$(N=*);!$(N#*)]")
        sulfonamide_pattern = Chem.MolFromSmarts("[S](=O)(=O)[N]")
        pentafluorophenol_pattern = Chem.MolFromSmarts("[O][c]1[c]([F])[c]([F])[c]([F])[c]([F])[c]1[F]")
        
        # Check reactants for activated sulfonate and amine
        has_activated_sulfonate = any(mol.HasSubstructMatch(activated_sulfonate_pattern) for mol in reactants)
        has_amine = any(mol.HasSubstructMatch(amine_pattern) for mol in reactants)
        
        # Check products for sulfonamide
        has_sulfonamide = any(mol.HasSubstructMatch(sulfonamide_pattern) for mol in products)
        
        # Check for pentafluorophenol leaving group in products
        has_leaving_group = any(mol.HasSubstructMatch(pentafluorophenol_pattern) for mol in products)
        
        # All conditions must be met for sulfonamide formation via activated ester
        return has_activated_sulfonate and has_amine and has_sulfonamide and has_leaving_group
