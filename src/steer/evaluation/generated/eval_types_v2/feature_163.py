"""Generated evaluation code for: Convergent synthesis via two complex fragments"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentPyrazoleFormation(BaseScoring):
    """
    Evaluates convergent synthesis strategy where two complex fragments 
    are coupled via pyrazole formation at a late stage.
    """
    
    def __init__(self, config: Dict):
        self.fragment_count = config.get("fragment_count", 2)
        self.coupling_reaction = config.get("coupling_reaction", "pyrazole_formation")
        self.convergence_stage = config.get("convergence_stage", "late")
        
        # SMARTS pattern for pyrazole formation reactions
        self.pyrazole_pattern = Chem.MolFromSmarts("c1cc[nH]n1")  # pyrazole core
        # Pattern for hydrazine precursor (common in pyrazole synthesis)
        self.hydrazine_pattern = Chem.MolFromSmarts("[NH2][NH2]")
        # Pattern for 1,3-dicarbonyl compounds (pyrazole precursor)
        self.dicarbonyl_pattern = Chem.MolFromSmarts("[#6](=[#8])[#6][#6](=[#8])")

    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Convergent pyrazole formation doesn't occur
        
        # Score based on convergence stage preference
        if self.convergence_stage == "late":
            # Penalize early convergence, reward late convergence
            return max(0, 10 * (1 - x))  # Higher score for later stage (smaller x)
        elif self.convergence_stage == "early":
            # Reward early convergence
            return max(0, 10 * x)
        else:  # "mid" stage
            # Optimal around 0.5 depth
            return max(0, 10 * (1 - 2 * abs(x - 0.5)))

    def hit_condition(self, d) -> bool:
        """
        Check if this reaction represents convergent pyrazole formation
        between two complex fragments.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            reactants_smiles, products_smiles = mapped_rxn.split(">>")
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            
            # Filter out None molecules
            reactants = [r for r in reactants if r is not None]
            products = [p for p in products if p is not None]
            
            if len(reactants) < self.fragment_count:
                return False
                
            # Check if products contain pyrazole
            has_pyrazole_product = any(p.HasSubstructMatch(self.pyrazole_pattern) for p in products)
            
            if not has_pyrazole_product:
                return False
                
            # Check if reactants don't have pyrazole (formation reaction)
            reactants_have_pyrazole = any(r.HasSubstructMatch(self.pyrazole_pattern) for r in reactants)
            
            if reactants_have_pyrazole:
                return False  # This is modification, not formation
                
            # Check for typical pyrazole formation patterns
            has_hydrazine = any(r.HasSubstructMatch(self.hydrazine_pattern) for r in reactants)
            has_dicarbonyl = any(r.HasSubstructMatch(self.dicarbonyl_pattern) for r in reactants)
            
            # Additional check: ensure fragments are "complex" (> 10 heavy atoms each)
            complex_fragments = [r for r in reactants if r.GetNumHeavyAtoms() > 10]
            
            if len(complex_fragments) >= self.fragment_count and (has_hydrazine or has_dicarbonyl):
                return True
                
            return False
            
        except Exception:
            return False
