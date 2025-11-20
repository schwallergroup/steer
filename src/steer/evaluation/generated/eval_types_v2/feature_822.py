"""Generated evaluation code for: Late stage Suzuki cross-coupling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageSuzukiCoupling(BaseScoring):
    """
    Evaluates whether a Suzuki cross-coupling reaction occurs in the late stages of synthesis.
    Suzuki coupling is used as the final step to install the biaryl moiety with high functional group tolerance.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "float")
        self.target_depth = config.get("target_depth", {}).get("value", 0.2)  # Default to early (late stage)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Suzuki coupling doesn't happen
        else:
            # Reward earlier depth values (late stage reactions)
            # Scale to 0-10 where 0 depth (last reaction) gives highest score
            return max(0, 10 * (1 - x))
    
    def hit_condition(self, d):
        """Check if this reaction is a Suzuki cross-coupling"""
        metadata = d.get("metadata", {})
        
        # Check policy name if available
        policy_name = metadata.get("policy_name", "")
        if "suzuki" in policy_name.lower():
            return True
        
        # Check reaction SMILES for Suzuki coupling pattern
        mapped_rxn = metadata.get("mapped_reaction_smiles")
        if not mapped_rxn:
            return False
        
        return self._is_suzuki_coupling(mapped_rxn)
    
    def _is_suzuki_coupling(self, reaction_smiles):
        """Detect Suzuki coupling by identifying boronic acid/ester consumption and biaryl formation"""
        try:
            rxn_parts = reaction_smiles.split(">>")
            if len(rxn_parts) != 2:
                return False
            
            products = rxn_parts[0]
            reactants = rxn_parts[1]
            
            # Parse molecules
            prod_mols = [Chem.MolFromSmiles(s.strip()) for s in products.split(".") if s.strip()]
            react_mols = [Chem.MolFromSmiles(s.strip()) for s in reactants.split(".") if s.strip()]
            
            if not all(prod_mols) or not all(react_mols):
                return False
            
            # Check for boronic acid/ester in reactants
            boronic_patterns = [
                "[B](O)(O)",  # Boronic acid
                "[B]1OC(C)(C)CO1",  # Pinacol boronate
                "[B](OC)(OC)",  # Boronic ester
            ]
            
            has_boron_reactant = any(
                any(react_mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)) 
                    for pattern in boronic_patterns)
                for react_mol in react_mols
            )
            
            # Check for aryl halide in reactants (Br, I, Cl on aromatic carbon)
            aryl_halide_patterns = [
                "c-Br",  # Aryl bromide
                "c-I",   # Aryl iodide  
                "c-Cl",  # Aryl chloride
            ]
            
            has_aryl_halide = any(
                any(react_mol.HasSubstructMatch(Chem.MolFromSmarts(pattern))
                    for pattern in aryl_halide_patterns)
                for react_mol in react_mols
            )
            
            # Check for biaryl formation in products
            biaryl_pattern = "c-c"  # Aromatic carbon-aromatic carbon bond
            has_biaryl_product = any(
                prod_mol.HasSubstructMatch(Chem.MolFromSmarts(biaryl_pattern))
                for prod_mol in prod_mols
            )
            
            return has_boron_reactant and has_aryl_halide and has_biaryl_product
            
        except Exception:
            return False
