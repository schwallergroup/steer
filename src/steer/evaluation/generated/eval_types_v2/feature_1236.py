"""Generated evaluation code for: Early Friedel-Crafts cyclization for core assembly"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlyFriedelCraftsCyclization(BaseScoring):
    """
    Evaluates whether an intramolecular Friedel-Crafts acylation reaction
    occurs early in the synthesis route for core ring assembly.
    """
    
    def __init__(self, config: Dict):
        self.timing = config["parameters"]["timing"]  # "early"
        self.intramolecular = config["parameters"]["intramolecular"]
    
    def route_scoring(self, x) -> float:
        """
        Score based on timing of Friedel-Crafts cyclization.
        Early occurrence (low depth fraction) gives higher score.
        """
        if x < 0:
            return 0  # Reaction doesn't occur
        else:
            # Early reactions get higher scores
            return 1 - x
    
    def hit_condition(self, d) -> bool:
        """
        Check if a reaction is an intramolecular Friedel-Crafts acylation
        that forms a ring.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            product = Chem.MolFromSmiles(rxn_parts[0])
            reactants = [Chem.MolFromSmiles(r) for r in rxn_parts[1].split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check if this is intramolecular (single reactant transforms to product)
            if self.intramolecular and len(reactants) > 1:
                # Filter out small molecules (catalysts, reagents)
                main_reactants = [r for r in reactants if r.GetNumAtoms() > 5]
                if len(main_reactants) > 1:
                    return False
            
            # Check for Friedel-Crafts acylation pattern
            if self._is_friedel_crafts_acylation(product, reactants):
                # Check if a ring was formed
                return self._ring_formed(product, reactants)
                
        except Exception:
            return False
            
        return False
    
    def _is_friedel_crafts_acylation(self, product, reactants) -> bool:
        """
        Detect Friedel-Crafts acylation by looking for aromatic C-CO bond formation.
        """
        # Pattern for aromatic carbon connected to carbonyl
        fc_pattern = Chem.MolFromSmarts("[cR]-[CX3](=O)")
        
        if not fc_pattern:
            return False
            
        # Product should have the Friedel-Crafts pattern
        if not product.HasSubstructMatch(fc_pattern):
            return False
            
        # Check if reactant had acyl halide or similar electrophile
        acyl_electrophile_patterns = [
            "[CX3](=O)[Cl,Br,I]",  # Acyl halide
            "[CX3](=O)O[CX3](=O)",  # Anhydride
            "[CX3](=O)[OH]"  # Carboxylic acid (with Lewis acid)
        ]
        
        has_electrophile = False
        for reactant in reactants:
            for pattern_smarts in acyl_electrophile_patterns:
                pattern = Chem.MolFromSmarts(pattern_smarts)
                if pattern and reactant.HasSubstructMatch(pattern):
                    has_electrophile = True
                    break
            if has_electrophile:
                break
                
        return has_electrophile
    
    def _ring_formed(self, product, reactants) -> bool:
        """
        Check if a ring was formed by comparing ring counts.
        """
        try:
            product_rings = product.GetRingInfo().NumRings()
            
            # Sum rings in all reactants
            reactant_rings = sum(r.GetRingInfo().NumRings() for r in reactants)
            
            # Ring formation should increase ring count
            return product_rings > reactant_rings
            
        except Exception:
            return False
