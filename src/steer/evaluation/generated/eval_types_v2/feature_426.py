"""Generated evaluation code for: Late stage nitrile to acetamido conversion"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class NitrileToAcetamidoConversion(BaseScoring):
    """
    Evaluates synthesis routes for late-stage conversion of nitrile to acetamido groups.
    Checks if a nitrile group is preserved through the synthesis and converted to 
    acetamido functionality at a late stage (preferably the final step).
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "float")
        self.target_depth = config.get("target_depth", {}).get("value", 1.0)
        
        # SMARTS patterns for nitrile and acetamido groups
        self.nitrile_pattern = Chem.MolFromSmarts("[C]#[N]")
        self.acetamido_pattern = Chem.MolFromSmarts("[N][C](=[O])[CH3]")
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Conversion doesn't happen
        else:
            # Late-stage conversion is better (closer to 1.0 = final step)
            if self.condition_type == "bool":
                return 1 if x >= 0.8 else 0  # Reward very late stage
            else:
                # Score higher for later conversions
                return 10 * x if x >= 0.5 else 2 * x
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction converts nitrile to acetamido group.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles")
        
        if not mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            product = Chem.MolFromSmiles(rxn_parts[0])
            reactants_smiles = rxn_parts[1].split(".")
            reactants = [Chem.MolFromSmiles(smi) for smi in reactants_smiles if smi]
            
            if not product or not reactants:
                return False
            
            # Check if product has acetamido group
            has_acetamido_product = product.HasSubstructMatch(self.acetamido_pattern)
            
            # Check if any reactant has nitrile group
            has_nitrile_reactant = any(
                reactant.HasSubstructMatch(self.nitrile_pattern) 
                for reactant in reactants if reactant
            )
            
            # Additional check: ensure nitrile is absent in product
            has_nitrile_product = product.HasSubstructMatch(self.nitrile_pattern)
            
            # This is a nitrile to acetamido conversion if:
            # 1. Reactant has nitrile
            # 2. Product has acetamido 
            # 3. Product doesn't have nitrile (conversion occurred)
            return (has_nitrile_reactant and 
                   has_acetamido_product and 
                   not has_nitrile_product)
                   
        except Exception:
            return False
