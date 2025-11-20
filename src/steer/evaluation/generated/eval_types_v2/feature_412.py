"""Generated evaluation code for: Boc protection for amide coupling selectivity"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BocProtectionStrategy(BaseScoring):
    """
    Evaluates synthesis routes for the use of Boc protection strategy on amines
    to enable selective amide coupling reactions. Checks if Boc protection occurs
    before amide coupling steps in the synthesis route.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "relative")
        self.target_depth = config.get("target_depth", {}).get("value", 0.3)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Boc protection strategy not found
        else:
            # Earlier use of Boc protection is better for selectivity
            if self.condition_type == "bool":
                return 1  # Strategy found
            else:
                return max(0, 1 - x)  # Penalize late-stage protection
    
    def hit_condition(self, d):
        """
        Check if this reaction involves Boc protection of an amine followed by
        potential for amide coupling selectivity.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants = rxn_parts[0]
            products = rxn_parts[1].split(".")
            
            # Check for Boc protection pattern
            boc_pattern = Chem.MolFromSmarts("[N;!$(N-C(=O)OC(C)(C)C)]-[#6]>>N-C(=O)OC(C)(C)C")
            tert_butyl_carbamate_pattern = Chem.MolFromSmarts("NC(=O)OC(C)(C)C")
            
            # Alternative: check for characteristic Boc reagent and amine substrate
            boc_anhydride_pattern = Chem.MolFromSmarts("CC(C)(C)OC(=O)OC(=O)OC(C)(C)C")
            amine_pattern = Chem.MolFromSmarts("[NX3;H2,H1;!$(NC=O)]")
            
            reactant_mol = Chem.MolFromSmiles(reactants)
            if not reactant_mol:
                return False
                
            # Check if reactants contain amine and Boc reagent
            has_amine = reactant_mol.HasSubstructMatch(amine_pattern)
            has_boc_reagent = any([
                reactant_mol.HasSubstructMatch(boc_anhydride_pattern),
                "BOC" in reactants.upper(),
                "tert-butyl" in reactants.lower()
            ])
            
            # Check if product contains Boc-protected amine
            has_boc_product = False
            for product_smiles in products:
                product_mol = Chem.MolFromSmiles(product_smiles)
                if product_mol and product_mol.HasSubstructMatch(tert_butyl_carbamate_pattern):
                    has_boc_product = True
                    break
            
            # Verify this is a protection reaction (amine + Boc reagent -> Boc-protected amine)
            if has_amine and (has_boc_reagent or has_boc_product):
                return True
                
            # Alternative check: look for explicit Boc protection in reaction metadata
            reaction_name = metadata.get("reaction_name", "").lower()
            if "boc" in reaction_name and ("protect" in reaction_name or "carbamate" in reaction_name):
                return True
                
        except Exception:
            return False
            
        return False
