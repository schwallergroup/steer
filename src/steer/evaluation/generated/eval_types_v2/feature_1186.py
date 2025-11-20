"""Generated evaluation code for: Multi-step reagent synthesis approach"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentStrategy(MultiRxnCondBase):
    """
    Evaluates whether a synthesis route uses a convergent strategy by checking
    if a commercially available reagent is synthesized through multiple steps
    instead of being purchased directly.
    """
    
    def __init__(self, config):
        self.reagent_synthesis_steps = config["parameters"]["reagent_synthesis_steps"]
        self.commercial_availability = config["parameters"]["commercial_availability"]
        self.reagent_name = config["parameters"]["reagent_name"]
        
        # Define SMARTS pattern for trityl chloride
        self.reagent_smarts = "ClC(c1ccccc1)(c2ccccc2)c3ccccc3"
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        """
        Check if the reagent is synthesized through the specified number of steps
        rather than used directly from commercial sources.
        """
        reactions = self.get_rxns(d)
        
        # Find reactions that produce the target reagent
        reagent_synthesis_depth = self.find_reagent_synthesis_depth(reactions)
        
        # Check if reagent is synthesized in approximately the target number of steps
        if reagent_synthesis_depth >= self.reagent_synthesis_steps:
            condition_met = True
        else:
            condition_met = False
            
        return condition_met, len(reactions)
    
    def find_reagent_synthesis_depth(self, reactions) -> int:
        """
        Find the depth at which the target reagent is synthesized.
        Returns the number of steps in the reagent synthesis pathway.
        """
        reagent_mol = Chem.MolFromSmarts(self.reagent_smarts)
        if reagent_mol is None:
            return 0
            
        reagent_synthesis_steps = 0
        
        for i, rxn_data in enumerate(reactions):
            if "mapped_reaction_smiles" not in rxn_data.get("metadata", {}):
                continue
                
            rxn_smiles = rxn_data["metadata"]["mapped_reaction_smiles"]
            rxn_parts = rxn_smiles.split(">>")
            
            if len(rxn_parts) != 2:
                continue
                
            # Check if the reagent is produced in this reaction
            products = rxn_parts[0].split(".")
            
            for prod_smiles in products:
                try:
                    prod_mol = Chem.MolFromSmiles(prod_smiles)
                    if prod_mol and prod_mol.HasSubstructMatch(reagent_mol):
                        # Found the reagent being synthesized
                        reagent_synthesis_steps = i + 1
                        break
                except:
                    continue
                    
        return reagent_synthesis_steps
    
    def route_scoring(self, x) -> float:
        """
        Score based on whether convergent strategy is employed.
        Higher scores for routes that synthesize the reagent through multiple steps.
        """
        if x < 0:
            return 0  # Condition not met
        else:
            # Reward convergent synthesis approach
            return 1.0  # Full score for using convergent strategy
