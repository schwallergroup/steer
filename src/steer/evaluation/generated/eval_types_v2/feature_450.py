"""Generated evaluation code for: Convergent synthesis via pyrrole-quinoline fragments"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentPyrroleQuinolineSynthesis(BaseScoring):
    """
    Evaluates convergent synthesis routes that join pyrrole and quinoline fragments
    via Suzuki cross-coupling at a specific step depth.
    """
    
    def __init__(self, config: Dict):
        self.fragment_count = config["parameters"]["fragment_count"]
        self.coupling_reaction = config["parameters"]["coupling_reaction"]
        self.target_coupling_step = config["parameters"]["coupling_step"]
        
        # Define SMARTS patterns for heterocyclic fragments
        self.pyrrole_pattern = Chem.MolFromSmarts("[cH]1[cH][cH][nH][cH]1")  # pyrrole ring
        self.quinoline_pattern = Chem.MolFromSmarts("c1ccc2ncccc2c1")  # quinoline ring
        
        # Suzuki coupling pattern (aryl-aryl bond formation)
        self.suzuki_pattern = Chem.MolFromSmarts("[c,C]-[c,C]")
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Convergent coupling not found
        
        # Score based on how close the coupling occurs to target step
        step_deviation = abs(x - (self.target_coupling_step / 10.0))  # Normalize to 0-1
        return max(0, 1 - step_deviation * 2)  # Penalize deviation from target step
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction represents convergent coupling of pyrrole-quinoline fragments"""
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
                
            product_smiles, reactants_smiles = rxn_smiles.split(">>")
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            
            if not product or len(reactants) < self.fragment_count:
                return False
            
            # Check if product contains both pyrrole and quinoline
            has_pyrrole = product.HasSubstructMatch(self.pyrrole_pattern)
            has_quinoline = product.HasSubstructMatch(self.quinoline_pattern)
            
            if not (has_pyrrole and has_quinoline):
                return False
            
            # Check if reactants contain the separate fragments
            pyrrole_reactants = []
            quinoline_reactants = []
            
            for reactant in reactants:
                if reactant.HasSubstructMatch(self.pyrrole_pattern):
                    pyrrole_reactants.append(reactant)
                if reactant.HasSubstructMatch(self.quinoline_pattern):
                    quinoline_reactants.append(reactant)
            
            # Verify convergent nature: separate fragments come together
            has_separate_fragments = len(pyrrole_reactants) >= 1 and len(quinoline_reactants) >= 1
            
            # Check for Suzuki-like coupling (simple heuristic: C-C bond formation)
            is_coupling_reaction = self._is_suzuki_coupling(product, reactants)
            
            return has_separate_fragments and is_coupling_reaction
            
        except Exception:
            return False
    
    def _is_suzuki_coupling(self, product, reactants) -> bool:
        """Heuristic check for Suzuki-type coupling reaction"""
        try:
            # Simple check: product has more C-C bonds than sum of reactants
            product_cc_bonds = len(product.GetSubstructMatches(self.suzuki_pattern))
            reactant_cc_bonds = sum(r.GetSubstructMatches(self.suzuki_pattern) 
                                   for r in reactants if r is not None)
            
            # Also check for typical Suzuki reagents/conditions in metadata
            metadata = {}  # Would check d.get("metadata", {}) for boronic acids, palladium, etc.
            
            return product_cc_bonds > reactant_cc_bonds
            
        except Exception:
            return False
