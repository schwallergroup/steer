"""Generated evaluation code for: Ketone to amine via oxime intermediate"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class KetoneToAmineViaOxime(MultiRxnCondBase):
    """
    Evaluates synthesis routes for ketone to amine conversion via oxime intermediate.
    Checks for the presence of both oxime formation and oxime reduction reactions
    in the correct sequence to transform a ketone to a primary amine.
    """
    
    def __init__(self, config):
        self.require_sequence = config.get("require_sequence", True)
        self.allow_partial = config.get("allow_partial", False)
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        oxime_formation_found = False
        oxime_reduction_found = False
        correct_sequence = False
        
        # Check each reaction for oxime formation or reduction
        for i, rxn in enumerate(reactions):
            if self.detect_oxime_formation(rxn):
                oxime_formation_found = True
                # Check if oxime reduction occurs later in the sequence
                if self.require_sequence:
                    for j in range(i+1, len(reactions)):
                        if self.detect_oxime_reduction(reactions[j]):
                            oxime_reduction_found = True
                            correct_sequence = True
                            break
            elif self.detect_oxime_reduction(rxn):
                oxime_reduction_found = True
        
        # Evaluate condition based on configuration
        if self.require_sequence:
            condition_met = correct_sequence
        elif self.allow_partial:
            condition_met = oxime_formation_found or oxime_reduction_found
        else:
            condition_met = oxime_formation_found and oxime_reduction_found
            
        return condition_met, len(reactions)
    
    def detect_oxime_formation(self, rxn):
        """
        Detects oxime formation: ketone + hydroxylamine -> oxime + water
        Looks for C=N-OH pattern formation from C=O pattern
        """
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        # Check for ketone pattern in reactants
        ketone_pattern = Chem.MolFromSmarts("[C:1]=[O:2]")
        hydroxylamine_pattern = Chem.MolFromSmarts("[N:3]-[OH:4]")
        
        # Check for oxime pattern in products
        oxime_pattern = Chem.MolFromSmarts("[C:1]=[N:3]-[OH:4]")
        
        try:
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".")]
            
            # Check reactants contain ketone and hydroxylamine-like species
            has_ketone = any(mol and mol.HasSubstructMatch(ketone_pattern) for mol in reactant_mols)
            has_nh2oh = any(mol and (mol.HasSubstructMatch(hydroxylamine_pattern) or 
                                   "N" in Chem.MolToSmiles(mol)) for mol in reactant_mols if mol)
            
            # Check products contain oxime
            has_oxime = any(mol and mol.HasSubstructMatch(oxime_pattern) for mol in product_mols)
            
            return has_ketone and has_nh2oh and has_oxime
            
        except:
            return False
    
    def detect_oxime_reduction(self, rxn):
        """
        Detects oxime reduction: oxime -> amine
        Looks for conversion of C=N-OH to C-NH2 or loss of C=N-OH pattern
        """
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        # Oxime pattern in reactants
        oxime_pattern = Chem.MolFromSmarts("[C:1]=[N:2]-[OH:3]")
        
        # Primary amine pattern in products
        primary_amine_pattern = Chem.MolFromSmarts("[C:1]-[NH2:2]")
        
        try:
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".")]
            
            # Check reactants contain oxime
            has_oxime_reactant = any(mol and mol.HasSubstructMatch(oxime_pattern) 
                                   for mol in reactant_mols)
            
            # Check products contain primary amine
            has_amine_product = any(mol and mol.HasSubstructMatch(primary_amine_pattern) 
                                  for mol in product_mols)
            
            # Additional check: oxime pattern should be absent in products
            oxime_consumed = not any(mol and mol.HasSubstructMatch(oxime_pattern) 
                                   for mol in product_mols)
            
            return has_oxime_reactant and has_amine_product and oxime_consumed
            
        except:
            return False
