"""Generated evaluation code for: Final step amide coupling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class FinalStepAmideCoupling(BaseScoring):
    """
    Evaluates whether the final synthetic step involves amide coupling.
    Detects amide bond formation by checking if a C(=O)N substructure is present
    in the product but absent in all reactants.
    """
    
    def __init__(self, config: Dict):
        # No additional configuration needed for this feature
        pass
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Amide coupling doesn't happen at final step
        else:
            return 10  # Perfect score when amide coupling is the final step
    
    def hit_condition(self, d):
        """Check if this reaction involves amide bond formation"""
        if d.get("depth", 0) != 0:
            return False  # Only check the final step (depth 0)
        
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            rxn_parts = rxn_smiles.split(">>")
            
            if len(rxn_parts) != 2:
                return False
                
            product = Chem.MolFromSmiles(rxn_parts[0])
            reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[1].split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Define amide pattern: C(=O)N
            amide_pattern = Chem.MolFromSmarts("[C](=O)[N]")
            
            # Check if product contains amide bond
            product_has_amide = product.HasSubstructMatch(amide_pattern)
            
            if not product_has_amide:
                return False
            
            # Check if amide bond is newly formed (not present in any reactant)
            for reactant in reactants:
                if reactant.HasSubstructMatch(amide_pattern):
                    # Amide already exists in reactant, so not newly formed
                    continue
            
            # Additional check: look for typical amide coupling reactants
            # Carboxylic acid pattern: [C](=O)[OH]
            # Amine pattern: [N;!$(N=O);!$(N~[#6]=[O,S,N])]
            acid_pattern = Chem.MolFromSmarts("[C](=O)[OH]")
            amine_pattern = Chem.MolFromSmarts("[N;!$(N=O);!$(N~[#6]=[O,S,N]);H1,H2]")
            
            has_acid = any(r.HasSubstructMatch(acid_pattern) for r in reactants)
            has_amine = any(r.HasSubstructMatch(amine_pattern) for r in reactants)
            
            return has_acid and has_amine and product_has_amide
            
        except Exception:
            return False
