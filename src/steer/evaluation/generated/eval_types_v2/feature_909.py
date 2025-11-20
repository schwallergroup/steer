"""Generated evaluation code for: Multiple sequential Williamson ether formations"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class MultipleSequentialWilliamsonEther(MultiRxnCondBase):
    """
    Evaluates routes for multiple sequential Williamson ether formations.
    Checks if the route contains exactly 3 consecutive Williamson ether synthesis reactions.
    """
    
    def __init__(self, config):
        self.required_count = config.get("count", 3)
        self.sequential = config.get("sequential", True)
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        williamson_reactions = []
        
        # Identify all Williamson ether reactions and their positions
        for i, rxn in enumerate(reactions):
            if self.detect_williamson_ether(rxn):
                williamson_reactions.append(i)
        
        if self.sequential:
            # Check for consecutive Williamson ether reactions
            condition = self.has_sequential_williamson(williamson_reactions)
        else:
            # Just check total count
            condition = len(williamson_reactions) >= self.required_count
            
        return condition, len(reactions)
    
    def detect_williamson_ether(self, rxn):
        """
        Detects Williamson ether synthesis by looking for:
        1. Formation of C-O-C ether bond
        2. Nucleophilic substitution pattern (alkoxide + alkyl halide)
        """
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants = rxn_parts[0].split(".")
            products = rxn_parts[1].split(".")
            
            # Look for ether formation pattern
            ether_pattern = Chem.MolFromSmarts("[C]-[O]-[C]")
            halide_pattern = Chem.MolFromSmarts("[C][Cl,Br,I]")
            alkoxide_pattern = Chem.MolFromSmarts("[C][O-]")
            
            # Check if product contains new ether bond
            product_has_ether = False
            for prod_smiles in products:
                try:
                    prod_mol = Chem.MolFromSmiles(prod_smiles)
                    if prod_mol and prod_mol.HasSubstructMatch(ether_pattern):
                        product_has_ether = True
                        break
                except:
                    continue
            
            if not product_has_ether:
                return False
            
            # Check if reactants have characteristic Williamson pattern
            has_halide = False
            has_alkoxide = False
            
            for react_smiles in reactants:
                try:
                    react_mol = Chem.MolFromSmiles(react_smiles)
                    if react_mol:
                        if react_mol.HasSubstructMatch(halide_pattern):
                            has_halide = True
                        if react_mol.HasSubstructMatch(alkoxide_pattern):
                            has_alkoxide = True
                except:
                    continue
            
            # Alternative pattern: alcohol + base + halide
            alcohol_pattern = Chem.MolFromSmarts("[C][OH]")
            has_alcohol = False
            
            if not has_alkoxide:
                for react_smiles in reactants:
                    try:
                        react_mol = Chem.MolFromSmiles(react_smiles)
                        if react_mol and react_mol.HasSubstructMatch(alcohol_pattern):
                            has_alcohol = True
                            break
                    except:
                        continue
            
            return has_halide and (has_alkoxide or has_alcohol)
            
        except Exception:
            return False
    
    def has_sequential_williamson(self, williamson_positions):
        """
        Checks if there are at least 'required_count' consecutive Williamson reactions.
        """
        if len(williamson_positions) < self.required_count:
            return False
        
        # Check for consecutive sequences
        for i in range(len(williamson_positions) - self.required_count + 1):
            consecutive = True
            for j in range(1, self.required_count):
                if williamson_positions[i + j] != williamson_positions[i] + j:
                    consecutive = False
                    break
            if consecutive:
                return True
        
        return False
