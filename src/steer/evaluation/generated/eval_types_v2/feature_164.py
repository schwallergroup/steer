"""Generated evaluation code for: Central pyrazole ring formation via Knorr synthesis"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class KnorrPyrazoleSynthesis(BaseScoring):
    """
    Evaluates synthesis routes for the presence of Knorr pyrazole synthesis.
    
    The Knorr pyrazole synthesis involves the condensation of a 1,3-dicarbonyl compound
    with a hydrazine derivative to form a pyrazole ring. This class detects when this
    specific ring formation occurs in the synthesis route.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "float")
        self.target_depth = config.get("target_depth", {}).get("value", 0.0)
    
    def route_scoring(self, x) -> float:
        """Convert depth fraction to score (0-10 scale)."""
        if x < 0:
            return 0  # Knorr synthesis doesn't occur
        else:
            # Earlier occurrence (lower depth) is better for central ring formation
            return max(0, 10 * (1 - x))
    
    def hit_condition(self, d) -> bool:
        """Check if a reaction represents Knorr pyrazole synthesis."""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            product_smiles, reactants_smiles = mapped_rxn.split(">>")
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r) for r in reactants_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check if pyrazole ring is formed in product
            pyrazole_pattern = Chem.MolFromSmarts("c1[nH]ncc1")  # Pyrazole ring
            if not product.HasSubstructMatch(pyrazole_pattern):
                return False
            
            # Check for characteristic Knorr synthesis pattern:
            # 1. At least one reactant should contain a 1,3-dicarbonyl or β-diketone
            # 2. At least one reactant should contain hydrazine (N-N bond)
            
            has_dicarbonyl = False
            has_hydrazine = False
            
            # Pattern for 1,3-dicarbonyl compounds (β-diketones, β-ketoesters)
            dicarbonyl_patterns = [
                Chem.MolFromSmarts("C(=O)CC(=O)"),  # β-diketone
                Chem.MolFromSmarts("C(=O)COC(=O)"),  # β-ketoester
                Chem.MolFromSmarts("O=C-C-C=O"),     # General 1,3-dicarbonyl
            ]
            
            # Pattern for hydrazine derivatives
            hydrazine_patterns = [
                Chem.MolFromSmarts("NN"),           # Hydrazine bond
                Chem.MolFromSmarts("N-N"),          # Hydrazine single bond
                Chem.MolFromSmarts("[NH2][NH2]"),   # Simple hydrazine
                Chem.MolFromSmarts("[NH2][NH]"),    # Substituted hydrazine
            ]
            
            for reactant in reactants:
                # Check for dicarbonyl component
                for pattern in dicarbonyl_patterns:
                    if reactant.HasSubstructMatch(pattern):
                        has_dicarbonyl = True
                        break
                
                # Check for hydrazine component
                for pattern in hydrazine_patterns:
                    if reactant.HasSubstructMatch(pattern):
                        has_hydrazine = True
                        break
            
            # Additional check: ensure pyrazole is not present in reactants
            pyrazole_in_reactants = any(r.HasSubstructMatch(pyrazole_pattern) for r in reactants)
            
            return has_dicarbonyl and has_hydrazine and not pyrazole_in_reactants
            
        except Exception:
            return False
