"""Generated evaluation code for: Multiple ether bond formations via Williamson synthesis"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class WilliamsonEtherSynthesis(MultiRxnCondBase):
    """
    Evaluates routes for multiple Williamson ether synthesis reactions.
    Specifically looks for phenol alkylation patterns occurring at least twice.
    """
    
    def __init__(self, config):
        self.required_count = config.get("count", 2)
        self.pattern_type = config.get("pattern", "phenol_alkylation")
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        williamson_count = sum(1 for r in reactions if self.detect_williamson_ether(r))
        
        condition = williamson_count >= self.required_count
        return condition, len(reactions)
    
    def detect_williamson_ether(self, rxn):
        """
        Detects Williamson ether synthesis by looking for:
        1. Formation of C-O-C ether bond
        2. Phenol + alkyl halide/tosylate pattern
        3. Loss of leaving group (Br, Cl, I, OTs)
        """
        try:
            prod_mol = Chem.MolFromSmiles(rxn[0])
            reactant_mols = [Chem.MolFromSmiles(r) for r in rxn[1].split(".")]
            
            # Look for phenol pattern in reactants
            phenol_pattern = Chem.MolFromSmarts("[OH1][c]")  # Phenolic OH
            alkyl_halide_pattern = Chem.MolFromSmarts("[CH2,CH1][Cl,Br,I]")  # Alkyl halide
            tosylate_pattern = Chem.MolFromSmarts("[CH2,CH1]OS(=O)(=O)[c]")  # Tosylate
            
            has_phenol = False
            has_alkylating_agent = False
            
            for reactant in reactant_mols:
                if reactant.HasSubstructMatch(phenol_pattern):
                    has_phenol = True
                if (reactant.HasSubstructMatch(alkyl_halide_pattern) or 
                    reactant.HasSubstructMatch(tosylate_pattern)):
                    has_alkylating_agent = True
            
            # Check if product has new ether linkage
            ether_pattern = Chem.MolFromSmarts("[c][OH0]([CH2,CH1])")  # Aryl-O-alkyl ether
            has_ether_product = prod_mol.HasSubstructMatch(ether_pattern)
            
            # Verify this is phenol alkylation specifically
            if self.pattern_type == "phenol_alkylation":
                return has_phenol and has_alkylating_agent and has_ether_product
            
            return has_ether_product and (has_phenol or has_alkylating_agent)
            
        except:
            return False
