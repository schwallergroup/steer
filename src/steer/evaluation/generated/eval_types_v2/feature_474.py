"""Generated evaluation code for: Three step aryl bromide to benzyl chloride conversion"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ArylBromideToBenzylChloride(MultiRxnCondBase):
    """
    Evaluates if a route contains a 3-step aryl bromide to benzyl chloride conversion
    via formylation -> reduction -> chlorination sequence.
    """
    
    def __init__(self, config):
        self.starting_group = config["parameters"]["starting_group"]  # "Ar-Br"
        self.ending_group = config["parameters"]["ending_group"]      # "Ar-CH2Cl"
        self.step_count = config["parameters"]["step_count"]          # 3
        self.intermediates = config["parameters"]["intermediates"]    # ["Ar-CHO", "Ar-CH2OH"]
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Define SMARTS patterns for functional groups
        aryl_br_pattern = "[#6]1:[#6]:[#6]:[#6]:[#6]:[#6]:1-[Br]"  # Aromatic C-Br
        aryl_cho_pattern = "[#6]1:[#6]:[#6]:[#6]:[#6]:[#6]:1-[CH1]=[O]"  # Aromatic aldehyde
        aryl_ch2oh_pattern = "[#6]1:[#6]:[#6]:[#6]:[#6]:[#6]:1-[CH2]-[OH]"  # Aromatic alcohol
        aryl_ch2cl_pattern = "[#6]1:[#6]:[#6]:[#6]:[#6]:[#6]:1-[CH2]-[Cl]"  # Aromatic chloride
        
        # Track the sequence through reactions
        sequence_found = False
        
        # Look for the 3-step sequence in consecutive reactions
        for i in range(len(reactions) - 2):
            # Check first step: Ar-Br -> Ar-CHO (formylation)
            step1_valid = self.check_transformation(reactions[i], aryl_br_pattern, aryl_cho_pattern)
            
            if step1_valid:
                # Check second step: Ar-CHO -> Ar-CH2OH (reduction)
                step2_valid = self.check_transformation(reactions[i+1], aryl_cho_pattern, aryl_ch2oh_pattern)
                
                if step2_valid:
                    # Check third step: Ar-CH2OH -> Ar-CH2Cl (chlorination)
                    step3_valid = self.check_transformation(reactions[i+2], aryl_ch2oh_pattern, aryl_ch2cl_pattern)
                    
                    if step3_valid:
                        sequence_found = True
                        break
        
        return sequence_found, len(reactions)
    
    def check_transformation(self, rxn, reactant_pattern, product_pattern):
        """
        Check if a reaction transforms a molecule with reactant_pattern to one with product_pattern.
        """
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants = rxn_parts[0].split(".")
            products = rxn_parts[1].split(".")
            
            # Check if any reactant has the starting pattern
            reactant_match = False
            for r_smi in reactants:
                mol = Chem.MolFromSmiles(r_smi)
                if mol and mol.HasSubstructMatch(Chem.MolFromSmarts(reactant_pattern)):
                    reactant_match = True
                    break
            
            if not reactant_match:
                return False
            
            # Check if any product has the ending pattern
            product_match = False
            for p_smi in products:
                mol = Chem.MolFromSmiles(p_smi)
                if mol and mol.HasSubstructMatch(Chem.MolFromSmarts(product_pattern)):
                    product_match = True
                    break
            
            return product_match
            
        except Exception:
            return False
    
    def route_scoring(self, x):
        """
        Score based on whether the 3-step sequence is found.
        x is the fraction of total reactions where condition is met.
        """
        if x > 0:
            return 10  # Sequence found
        else:
            return 0   # Sequence not found
