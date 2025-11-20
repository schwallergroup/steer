"""Generated evaluation code for: Late N-glycosylation with methyl glycoside donor"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateNGlycosylation(BaseScoring):
    """
    Evaluates whether N-glycosylation occurs late in the synthesis using a methyl glycoside donor.
    Detects the formation of N-glycosidic bonds where a methyl glycoside acts as the glycosyl donor.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "float")
        self.target_depth = config.get("target_depth", {}).get("value", 0.8)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # N-glycosylation doesn't happen
        else:
            return 1 - x  # Late-stage N-glycosylation is better
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction represents N-glycosylation with methyl glycoside donor"""
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
                
            reactants_smiles, product_smiles = rxn_smiles.split(">>")
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            product = Chem.MolFromSmiles(product_smiles.strip())
            
            if not all(reactants) or not product:
                return False
            
            # Check if N-glycosidic bond is formed
            if not self._has_n_glycosidic_bond_formation(reactants, product):
                return False
            
            # Check if methyl glycoside donor is present
            if not self._has_methyl_glycoside_donor(reactants):
                return False
                
            return True
            
        except Exception:
            return False
    
    def _has_n_glycosidic_bond_formation(self, reactants, product) -> bool:
        """Check if an N-glycosidic bond (C-O-C-N pattern) is formed"""
        # N-glycosidic bond pattern: sugar carbon linked to nitrogen through oxygen
        n_glycosidic_pattern = Chem.MolFromSmarts("[CH1,CH2][O][CH1,CH2][NH1,NH2,N]")
        
        if not n_glycosidic_pattern:
            return False
        
        # Check if product has N-glycosidic bond
        if not product.HasSubstructMatch(n_glycosidic_pattern):
            return False
        
        # Check that this bond is newly formed (not present in any reactant)
        for reactant in reactants:
            if reactant.HasSubstructMatch(n_glycosidic_pattern):
                return False
                
        return True
    
    def _has_methyl_glycoside_donor(self, reactants) -> bool:
        """Check if one of the reactants is a methyl glycoside donor"""
        # Methyl glycoside pattern: methoxy group attached to anomeric carbon
        # [CH1,CH2] represents anomeric carbon, [O] is glycosidic oxygen, [CH3] is methyl
        methyl_glycoside_pattern = Chem.MolFromSmarts("[CH1,CH2][O][CH3]")
        
        if not methyl_glycoside_pattern:
            return False
        
        # Also look for common sugar ring patterns with methyl glycoside
        pyranose_methyl_pattern = Chem.MolFromSmarts("[CH1]1[CH1,CH2][CH1][CH1][CH1][O]1[O][CH3]")
        furanose_methyl_pattern = Chem.MolFromSmarts("[CH1]1[CH1,CH2][CH1][CH1][O]1[O][CH3]")
        
        for reactant in reactants:
            if (reactant.HasSubstructMatch(methyl_glycoside_pattern) or 
                (pyranose_methyl_pattern and reactant.HasSubstructMatch(pyranose_methyl_pattern)) or
                (furanose_methyl_pattern and reactant.HasSubstructMatch(furanose_methyl_pattern))):
                return True
                
        return False
