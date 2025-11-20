"""Generated evaluation code for: Late pyrazole ring formation via Claisen-cyclization sequence"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageClaisenPyrazoleFormation(BaseScoring):
    """
    Evaluates routes for late-stage pyrazole ring formation via Claisen-cyclization sequence.
    Checks if pyrazole rings are formed through Claisen condensation followed by cyclization,
    with preference for late-stage formation.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]  # "c1ccnnc1"
        self.timing = config["parameters"]["timing"]  # "late" 
        self.formation_method = config["parameters"]["formation_method"]  # "claisen_condensation_cyclization"
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Pyrazole formation doesn't happen
        else:
            # Late-stage formation is preferred (lower depth fraction is better)
            return 1 - x  # Convert to 0-1 score where late = high score
            
    def hit_condition(self, d) -> bool:
        """Check if this reaction forms a pyrazole ring via Claisen-cyclization"""
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
                
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            
            if not all(reactants) or not all(products):
                return False
                
            # Check if pyrazole ring is formed (present in products but not reactants)
            pyrazole_pattern = Chem.MolFromSmarts(self.ring_smarts)
            if not pyrazole_pattern:
                return False
                
            # Count pyrazole rings in reactants vs products
            reactant_pyrazoles = sum(mol.HasSubstructMatch(pyrazole_pattern) for mol in reactants)
            product_pyrazoles = sum(mol.HasSubstructMatch(pyrazole_pattern) for mol in products)
            
            # Pyrazole must be formed (more in products than reactants)
            if product_pyrazoles <= reactant_pyrazoles:
                return False
                
            # Check for Claisen condensation pattern (C-C bond formation with carbonyl)
            claisen_indicators = [
                "[C:1](=[O:2])[CH2:3]",  # Active methylene
                "[C:1](=[O:2])[O:3][C:4]",  # Ester functionality
                "[NH:1][NH:2]",  # Hydrazine for cyclization
            ]
            
            has_claisen_features = False
            for pattern_smarts in claisen_indicators:
                pattern = Chem.MolFromSmarts(pattern_smarts)
                if pattern and any(mol.HasSubstructMatch(pattern) for mol in reactants):
                    has_claisen_features = True
                    break
                    
            return has_claisen_features
            
        except Exception:
            return False
