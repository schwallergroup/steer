"""Generated evaluation code for: Late proline ring formation via double alkylation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateProlineRingFormation(BaseScoring):
    """
    Evaluates synthesis routes for late-stage proline ring formation via double alkylation.
    Checks if a proline ring (C1CCCN1) is formed through double alkylation reactions
    and rewards later formation in the synthesis route.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]  # "C1CCCN1"
        self.timing = config["parameters"]["timing"]  # "late"
        self.formation_method = config["parameters"]["formation_method"]  # "double_alkylation"
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        """Convert depth fraction to score, rewarding late-stage formation"""
        if x < 0:
            return 0  # Ring formation doesn't happen
        else:
            # Late-stage formation gets higher score (closer to 1.0 depth gets higher score)
            return 10 * x  # Scale to 0-10 range, later is better
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction forms a proline ring via double alkylation"""
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            rxn_parts = rxn_smiles.split(">>")
            
            if len(rxn_parts) != 2:
                return False
                
            reactants_smiles = rxn_parts[0]
            products_smiles = rxn_parts[1]
            
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            # Check if any product has the proline ring that wasn't in reactants
            product_has_ring = any(
                mol and mol.HasSubstructMatch(self.ring_pattern) 
                for mol in products
            )
            
            reactant_has_ring = any(
                mol and mol.HasSubstructMatch(self.ring_pattern) 
                for mol in reactants
            )
            
            # Ring must be formed (present in products but not reactants)
            if not (product_has_ring and not reactant_has_ring):
                return False
            
            # Check for double alkylation pattern
            return self._is_double_alkylation(reactants, products)
            
        except Exception:
            return False
    
    def _is_double_alkylation(self, reactants, products) -> bool:
        """Check if the reaction represents a double alkylation mechanism"""
        # Look for primary amine in reactants and dibromo/dichloro compound
        has_primary_amine = False
        has_dihalide = False
        
        # Patterns for detection
        primary_amine_pattern = Chem.MolFromSmarts("[NH2]")
        dibromo_pattern = Chem.MolFromSmarts("BrCCCCBr")  # Dibromo butane for proline
        dichloro_pattern = Chem.MolFromSmarts("ClCCCCCl")  # Alternative dihalide
        
        for mol in reactants:
            if mol:
                # Check for primary amine
                if mol.HasSubstructMatch(primary_amine_pattern):
                    has_primary_amine = True
                
                # Check for dihalide (dibromo or dichloro butane chain)
                if (mol.HasSubstructMatch(dibromo_pattern) or 
                    mol.HasSubstructMatch(dichloro_pattern)):
                    has_dihalide = True
        
        # Alternative check: look for any 4-carbon dihalide chain
        if not has_dihalide:
            general_dihalide = Chem.MolFromSmarts("[Br,Cl]CCCC[Br,Cl]")
            has_dihalide = any(
                mol and mol.HasSubstructMatch(general_dihalide) 
                for mol in reactants
            )
        
        return has_primary_amine and has_dihalide
