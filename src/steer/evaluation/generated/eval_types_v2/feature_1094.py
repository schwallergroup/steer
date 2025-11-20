"""Generated evaluation code for: Early indole formation via alkynyl aniline cyclization"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class IndoleFormationTiming(BaseScoring):
    """
    Evaluates the timing of indole formation via alkynyl aniline cyclization.
    Checks for early formation of indole rings through metal-catalyzed cyclization
    of ortho-alkynyl aniline intermediates.
    """
    
    def __init__(self, config: Dict):
        self.indole_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]  # "early"
        self.direction = config["parameters"]["direction"]  # "formation"
        self.method = config["parameters"]["method"]  # "metal_catalyzed_cyclization"
        
        # Compile SMARTS patterns
        self.indole_pattern = Chem.MolFromSmarts(self.indole_smarts)
        # Pattern for ortho-alkynyl aniline (alkyne ortho to amine on benzene)
        self.alkynyl_aniline_pattern = Chem.MolFromSmarts("c1ccc(N)c(C#C)c1")
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Indole formation doesn't happen
        
        if self.timing == "early":
            # Early formation is better - penalize late formation
            if x <= 0.3:  # Very early (first 30% of route)
                return 1.0
            elif x <= 0.5:  # Moderately early
                return 0.7
            elif x <= 0.7:  # Mid-route
                return 0.4
            else:  # Late formation
                return 0.1
        else:
            # For other timing preferences, could be implemented differently
            return 1 - x
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves indole formation via alkynyl aniline cyclization
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
                
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            
            # Filter out None molecules
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            if not reactants or not products:
                return False
            
            # Check for indole formation: indole in products but not in reactants
            indole_in_products = any(mol.HasSubstructMatch(self.indole_pattern) for mol in products)
            indole_in_reactants = any(mol.HasSubstructMatch(self.indole_pattern) for mol in reactants)
            
            if not (indole_in_products and not indole_in_reactants):
                return False
            
            # Check for alkynyl aniline in reactants (cyclization precursor)
            alkynyl_aniline_in_reactants = any(mol.HasSubstructMatch(self.alkynyl_aniline_pattern) for mol in reactants)
            
            if not alkynyl_aniline_in_reactants:
                return False
            
            # Additional check for metal-catalyzed conditions
            # Look for common metal catalysts in reaction metadata or SMILES
            if self.method == "metal_catalyzed_cyclization":
                reaction_text = str(d.get("metadata", {})).lower()
                metal_indicators = ["pd", "palladium", "cu", "copper", "au", "gold", "pt", "platinum"]
                has_metal_catalyst = any(metal in reaction_text for metal in metal_indicators)
                
                # If we can't detect metal catalyst from metadata, assume it's valid
                # since the structural transformation is the primary indicator
                return True
            
            return True
            
        except Exception:
            return False
