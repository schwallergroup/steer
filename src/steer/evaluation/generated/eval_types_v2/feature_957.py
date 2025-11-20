"""Generated evaluation code for: Early Williamson ether synthesis timing"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class WilliamsonEtherTiming(BaseScoring):
    """
    Evaluates whether Williamson ether synthesis occurs early in the synthesis route.
    
    Detects C-O bond formation between phenoxide/alkoxide and alkyl halide/tosylate,
    rewarding early-stage ether formation (depth < stage_threshold).
    """
    
    def __init__(self, config: Dict):
        self.stage_threshold = config["parameters"].get("stage_threshold", 0.2)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Williamson ether synthesis doesn't occur
        else:
            # Early stage is better - reward if before threshold
            if x <= self.stage_threshold:
                return 10 * (1 - x / self.stage_threshold)  # Scale 10-0 within early stage
            else:
                return 0  # Late stage gets no reward
                
    def hit_condition(self, d) -> bool:
        """Detect Williamson ether synthesis in a reaction"""
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            reactants_smiles, product_smiles = rxn_smiles.split(">>")
            
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
                
            # Check for ether formation: C-O-C where O connects two carbons
            return self._is_williamson_ether_formation(reactants, product)
            
        except:
            return False
            
    def _is_williamson_ether_formation(self, reactants, product):
        """Check if reaction involves phenoxide/alkoxide + alkyl halide -> ether"""
        
        # Pattern for phenoxide/alkoxide (oxygen with negative charge or connected to aromatic)
        phenoxide_pattern = Chem.MolFromSmarts("[O-,OH]-[c,C]")
        # Pattern for alkyl halides/tosylates
        alkyl_halide_pattern = Chem.MolFromSmarts("[C]-[Br,Cl,I,F]")
        tosylate_pattern = Chem.MolFromSmarts("[C]-[O]-S(=O)(=O)-[c]")
        
        # Pattern for ether product
        ether_pattern = Chem.MolFromSmarts("[c,C]-[O]-[C]")
        
        if not product.HasSubstructMatch(ether_pattern):
            return False
            
        # Check if reactants contain nucleophile (phenoxide/alkoxide) and electrophile
        has_nucleophile = False
        has_electrophile = False
        
        for reactant in reactants:
            if reactant.HasSubstructMatch(phenoxide_pattern):
                has_nucleophile = True
            if (reactant.HasSubstructMatch(alkyl_halide_pattern) or 
                reactant.HasSubstructMatch(tosylate_pattern)):
                has_electrophile = True
                
        return has_nucleophile and has_electrophile
