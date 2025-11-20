"""Generated evaluation code for: Late stage triazole N-alkylation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageTriazoleNAlkylation(BaseScoring):
    """
    Evaluates whether triazole N-alkylation occurs at late stages of synthesis.
    Returns higher scores for earlier occurrence of the reaction to penalize 
    late-stage triazole N-alkylation which can cause regioselectivity issues.
    """
    
    def __init__(self, config: Dict):
        self.timing_penalty = config.get("timing_penalty", 1.0)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Reaction doesn't occur
        else:
            # Penalize late-stage occurrence (higher depth fraction = later stage)
            # Early stage (low x) gets higher score, late stage (high x) gets lower score
            return 1.0 - x
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves N-alkylation of a triazole ring.
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
            
            # Check if product contains triazole
            triazole_patterns = [
                "[#7]1-[#7]=[#7]-[#6]=[#6]-1",  # 1,2,3-triazole
                "[#7]1=[#7]-[#6]=[#6]-[#7]-1",  # 1,2,4-triazole
                "[#7]1-[#7]=[#6]-[#7]=[#6]-1"   # 1,3,5-triazine (alternative)
            ]
            
            product_has_triazole = any(
                product.HasSubstructMatch(Chem.MolFromSmarts(pattern))
                for pattern in triazole_patterns
            )
            
            if not product_has_triazole:
                return False
            
            # Check if any reactant has triazole with unsubstituted nitrogen
            unsubstituted_triazole_patterns = [
                "[#7H]1-[#7]=[#7]-[#6]=[#6]-1",  # 1,2,3-triazole with NH
                "[#7H]1=[#7]-[#6]=[#6]-[#7]-1",  # 1,2,4-triazole with NH
                "[#7]1-[#7H]=[#7]-[#6]=[#6]-1",  # 1,2,3-triazole with NH at different position
                "[#7]1=[#7]-[#6]=[#6]-[#7H]-1"   # 1,2,4-triazole with NH at different position
            ]
            
            reactant_has_unsubst_triazole = any(
                any(reactant.HasSubstructMatch(Chem.MolFromSmarts(pattern))
                    for pattern in unsubstituted_triazole_patterns)
                for reactant in reactants
            )
            
            if not reactant_has_unsubst_triazole:
                return False
            
            # Check for presence of alkylating agent (alkyl halide or similar)
            alkylating_patterns = [
                "[#6]-[Cl,Br,I]",  # alkyl halides
                "[#6]-O-S(=O)(=O)-[#6]",  # alkyl sulfonates (tosylates, mesylates)
                "[#6]-O-S(=O)(=O)-[F]"    # alkyl triflates
            ]
            
            has_alkylating_agent = any(
                any(reactant.HasSubstructMatch(Chem.MolFromSmarts(pattern))
                    for pattern in alkylating_patterns)
                for reactant in reactants
            )
            
            return has_alkylating_agent
            
        except Exception:
            return False
