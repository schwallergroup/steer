"""Generated evaluation code for: Commercial reagent multi-step synthesis"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class CommercialReagentSynthesis(MultiRxnCondBase):
    """
    Evaluates routes that synthesize commercially available reagents instead of purchasing them.
    Checks if tri-n-butyltin hydride is synthesized through multiple steps when it's commercially available.
    """
    
    def __init__(self, config):
        self.reagent_name = config["reagent_name"]
        self.synthesis_steps = config["synthesis_steps"]
        self.commercial_availability = config["commercial_availability"]
        
        # SMARTS pattern for tri-n-butyltin hydride (Bu3SnH)
        self.reagent_smarts = "[Sn]([CH2][CH2][CH2][CH3])([CH2][CH2][CH2][CH3])([CH2][CH2][CH2][CH3])[H]"
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        """Check if the route synthesizes the commercial reagent through multiple steps"""
        reactions = self.get_rxns(d)
        total_steps = len(reactions)
        
        # Check if target reagent appears as a product in any reaction
        synthesizes_reagent = any(self.produces_target_reagent(r) for r in reactions)
        
        # Check if reagent is used as reactant in subsequent reactions
        uses_synthesized_reagent = any(self.uses_target_reagent(r) for r in reactions)
        
        # Condition met if: reagent is synthesized, used in route, and route has expected steps
        condition = (synthesizes_reagent and 
                    uses_synthesized_reagent and 
                    total_steps >= self.synthesis_steps and
                    self.commercial_availability)
        
        return condition, total_steps
    
    def produces_target_reagent(self, rxn):
        """Check if reaction produces the target reagent"""
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            products = rxn_parts[1].split(".")
            pattern_mol = Chem.MolFromSmarts(self.reagent_smarts)
            
            if pattern_mol is None:
                return False
                
            for product_smiles in products:
                product_mol = Chem.MolFromSmiles(product_smiles)
                if product_mol and product_mol.HasSubstructMatch(pattern_mol):
                    return True
                    
        except Exception:
            pass
            
        return False
    
    def uses_target_reagent(self, rxn):
        """Check if reaction uses the target reagent as a reactant"""
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants = rxn_parts[0].split(".")
            pattern_mol = Chem.MolFromSmarts(self.reagent_smarts)
            
            if pattern_mol is None:
                return False
                
            for reactant_smiles in reactants:
                reactant_mol = Chem.MolFromSmiles(reactant_smiles)
                if reactant_mol and reactant_mol.HasSubstructMatch(pattern_mol):
                    return True
                    
        except Exception:
            pass
            
        return False
