"""Generated evaluation code for: Convergent synthesis via two complex fragments"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentFischerIndole(BaseScoring):
    """
    Evaluates convergent synthesis via Fischer indole formation from two complex fragments.
    Checks for the presence of Fischer indole reaction (hydrazine + ketone -> indole) and
    evaluates whether it occurs at the appropriate convergence stage.
    """
    
    def __init__(self, config: Dict):
        self.fragment_count = config.get("fragment_count", 2)
        self.convergence_stage = config.get("convergence_stage", "late")  # "early", "mid", "late"
        
        # Fischer indole reaction patterns
        self.hydrazine_pattern = "[NH2][NH]"  # Hydrazine component
        self.ketone_pattern = "[CX3]=[OX1]"   # Carbonyl component
        self.indole_product = "c1ccc2[nH]ccc2c1"  # Indole core
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Fischer indole reaction not found
        
        # Score based on convergence stage preference
        if self.convergence_stage == "late":
            return max(0, 10 * (1 - x))  # Prefer later in synthesis (lower depth fraction)
        elif self.convergence_stage == "early":
            return max(0, 10 * x)  # Prefer earlier in synthesis (higher depth fraction)
        else:  # mid
            return max(0, 10 * (1 - abs(x - 0.5) * 2))  # Prefer middle of synthesis
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction is a Fischer indole synthesis."""
        metadata = d.get("metadata", {})
        if "mapped_reaction_smiles" not in metadata:
            return False
            
        rxn_smiles = metadata["mapped_reaction_smiles"]
        rxn_parts = rxn_smiles.split(">>")
        
        if len(rxn_parts) != 2:
            return False
            
        product_smiles = rxn_parts[0]
        reactant_smiles = rxn_parts[1]
        
        try:
            # Parse molecules
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactant_smiles.split(".")]
            
            if not product or len(reactants) < 2:
                return False
            
            # Check if product contains indole core
            indole_query = Chem.MolFromSmarts(self.indole_product)
            if not product.HasSubstructMatch(indole_query):
                return False
            
            # Check if reactants contain hydrazine and ketone components
            hydrazine_query = Chem.MolFromSmarts(self.hydrazine_pattern)
            ketone_query = Chem.MolFromSmarts(self.ketone_pattern)
            
            has_hydrazine = False
            has_ketone = False
            complex_fragments = 0
            
            for reactant in reactants:
                if reactant.HasSubstructMatch(hydrazine_query):
                    has_hydrazine = True
                if reactant.HasSubstructMatch(ketone_query):
                    has_ketone = True
                
                # Count as complex fragment if it has more than 6 heavy atoms
                if reactant.GetNumHeavyAtoms() > 6:
                    complex_fragments += 1
            
            # Fischer indole conditions:
            # 1. Must have both hydrazine and ketone components
            # 2. Must form indole product
            # 3. Should involve the specified number of complex fragments
            return (has_hydrazine and has_ketone and 
                   complex_fragments >= self.fragment_count)
                   
        except Exception:
            return False
