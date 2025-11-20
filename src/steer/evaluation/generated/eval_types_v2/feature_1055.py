"""Generated evaluation code for: Late stage phenol ether formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStagePhenyletherFormation(BaseScoring):
    """
    Evaluates routes for late-stage phenol ether formation via Williamson ether synthesis.
    Penalizes routes that attempt phenol O-alkylation as a final step due to potential
    acid-base competition issues.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "bool")
        self.target_depth = config.get("target_depth", {}).get("value", -1)
        self.phenol_pattern = Chem.MolFromSmarts("[OH]c1ccccc1")
        
    def route_scoring(self, x) -> float:
        if self.condition_type == "bool":
            if self.target_depth == -1:  # Penalize if condition is met (late stage)
                return 0 if x >= 0 else 1
            else:
                return 1 if x >= 0 else 0
        else:
            if x < 0:
                return 1  # Good - no late stage phenol ether formation
            # Penalize based on how late the reaction occurs
            return max(0, 1 - (1 - x))  # Earlier is better
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves phenol ether formation via Williamson synthesis.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            product = Chem.MolFromSmiles(rxn_parts[0])
            reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[1].split(".")]
            
            if not product or not all(reactants):
                return False
                
            # Check if product has phenol ether (phenol pattern bonded to carbon via oxygen)
            phenol_ether_pattern = Chem.MolFromSmarts("c1ccccc1O[CH2,CH3]")
            if not product.HasSubstructMatch(phenol_ether_pattern):
                return False
                
            # Check if any reactant has free phenol group
            has_phenol_reactant = any(
                reactant.HasSubstructMatch(self.phenol_pattern) 
                for reactant in reactants
            )
            
            # Check if any reactant has alkyl halide or tosylate pattern (typical Williamson reagents)
            alkyl_halide_pattern = Chem.MolFromSmarts("[CH2,CH3][Cl,Br,I]")
            tosylate_pattern = Chem.MolFromSmarts("[CH2,CH3]OS(=O)(=O)c1ccc(C)cc1")
            
            has_alkylating_agent = any(
                reactant.HasSubstructMatch(alkyl_halide_pattern) or 
                reactant.HasSubstructMatch(tosylate_pattern)
                for reactant in reactants
            )
            
            return has_phenol_reactant and has_alkylating_agent
            
        except Exception:
            return False
