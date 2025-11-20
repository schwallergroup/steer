"""Generated evaluation code for: Two-step Cbz protecting group installation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class CbzTwoStepInstallation(MultiRxnCondBase):
    """
    Evaluates synthesis routes for two-step Cbz protecting group installation
    via carbamoyl chloride intermediate using phosgene and benzyl alcohol.
    """
    
    def __init__(self, config):
        self.required_steps = config.get("installation_steps", 2)
        self.target_protecting_group = config.get("protecting_group", "Cbz")
        self.required_reagents = set(config.get("reagents", ["phosgene", "benzyl_alcohol"]))
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Find Cbz installation sequence
        cbz_sequence = self.find_cbz_installation_sequence(reactions)
        
        if not cbz_sequence:
            return False, len(reactions)
            
        # Check if it's a two-step process with correct reagents
        is_two_step = len(cbz_sequence) == self.required_steps
        has_correct_reagents = self.verify_reagents(cbz_sequence)
        has_intermediate = self.verify_carbamoyl_chloride_intermediate(cbz_sequence)
        
        condition = is_two_step and has_correct_reagents and has_intermediate
        return condition, len(reactions)
    
    def find_cbz_installation_sequence(self, reactions):
        """Find sequence of reactions that install Cbz protecting group"""
        cbz_reactions = []
        
        for i, rxn in enumerate(reactions):
            if self.involves_cbz_chemistry(rxn):
                cbz_reactions.append((i, rxn))
        
        # Sort by reaction order and return consecutive sequences
        if len(cbz_reactions) >= 2:
            # Check for consecutive reactions
            for j in range(len(cbz_reactions) - 1):
                if cbz_reactions[j+1][0] - cbz_reactions[j][0] == 1:
                    return [cbz_reactions[j][1], cbz_reactions[j+1][1]]
        
        return []
    
    def involves_cbz_chemistry(self, rxn):
        """Check if reaction involves Cbz group formation or intermediate"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        # Cbz protecting group pattern
        cbz_pattern = Chem.MolFromSmarts("NC(=O)OCc1ccccc1")
        
        # Carbamoyl chloride intermediate pattern
        carbamoyl_pattern = Chem.MolFromSmarts("NC(=O)Cl")
        
        try:
            prod_mol = Chem.MolFromSmiles(products)
            react_mol = Chem.MolFromSmiles(reactants)
            
            if prod_mol and cbz_pattern:
                if prod_mol.HasSubstructMatch(cbz_pattern):
                    return True
                    
            if react_mol and carbamoyl_pattern:
                if react_mol.HasSubstructMatch(carbamoyl_pattern):
                    return True
                    
        except:
            pass
            
        return False
    
    def verify_reagents(self, reaction_sequence):
        """Verify that phosgene and benzyl alcohol are used in the sequence"""
        found_reagents = set()
        
        phosgene_patterns = [
            "ClC(=O)Cl",  # phosgene
            "O=C(Cl)Cl"   # alternative representation
        ]
        
        benzyl_alcohol_patterns = [
            "OCc1ccccc1",  # benzyl alcohol
            "c1ccc(CO)cc1" # alternative representation
        ]
        
        for rxn in reaction_sequence:
            reactants = rxn.split(">>")[0]
            
            # Check for phosgene
            for pattern in phosgene_patterns:
                if pattern in reactants:
                    found_reagents.add("phosgene")
                    break
            
            # Check for benzyl alcohol
            for pattern in benzyl_alcohol_patterns:
                if pattern in reactants:
                    found_reagents.add("benzyl_alcohol")
                    break
        
        return self.required_reagents.issubset(found_reagents)
    
    def verify_carbamoyl_chloride_intermediate(self, reaction_sequence):
        """Verify carbamoyl chloride intermediate is formed and consumed"""
        if len(reaction_sequence) != 2:
            return False
            
        first_rxn = reaction_sequence[0]
        second_rxn = reaction_sequence[1]
        
        # Check if first reaction produces carbamoyl chloride
        first_products = first_rxn.split(">>")[1]
        carbamoyl_pattern = Chem.MolFromSmarts("NC(=O)Cl")
        
        try:
            first_prod_mol = Chem.MolFromSmiles(first_products)
            if first_prod_mol and carbamoyl_pattern:
                has_intermediate = first_prod_mol.HasSubstructMatch(carbamoyl_pattern)
                
                # Check if second reaction consumes it to form Cbz
                second_reactants = second_rxn.split(">>")[0]
                second_products = second_rxn.split(">>")[1]
                
                second_react_mol = Chem.MolFromSmiles(second_reactants)
                second_prod_mol = Chem.MolFromSmiles(second_products)
                
                if second_react_mol and second_prod_mol:
                    consumes_intermediate = second_react_mol.HasSubstructMatch(carbamoyl_pattern)
                    cbz_pattern = Chem.MolFromSmarts("NC(=O)OCc1ccccc1")
                    forms_cbz = second_prod_mol.HasSubstructMatch(cbz_pattern)
                    
                    return has_intermediate and consumes_intermediate and forms_cbz
                    
        except:
            pass
            
        return False
