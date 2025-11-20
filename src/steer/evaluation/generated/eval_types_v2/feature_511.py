"""Generated evaluation code for: Early stage piperidine ring formation via alkylation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlyPiperidineAlkylation(BaseScoring):
    """
    Evaluates whether piperidine ring formation via double alkylation occurs early in the synthesis.
    Rewards routes where piperidine formation happens within the first 5 steps from the start.
    """
    
    def __init__(self, config: Dict):
        self.step_position_from_start = config["parameters"]["step_position_from_start"]
        self.timing = config["parameters"]["timing"]
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Condition not met
        
        # For early timing, reward smaller depth values (closer to start)
        if self.timing == "early":
            if x <= self.step_position_from_start / 10.0:  # Within target early range
                return 10 * (1 - x)  # Higher score for earlier occurrence
            else:
                return max(0, 5 * (1 - x))  # Reduced score for later occurrence
        
        return 0
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction involves piperidine ring formation via double alkylation"""
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            
            # Remove None molecules
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            if not reactants or not products:
                return False
            
            # Check for piperidine formation
            return self._is_piperidine_double_alkylation(reactants, products)
            
        except Exception:
            return False
    
    def _is_piperidine_double_alkylation(self, reactants, products) -> bool:
        """
        Check if the reaction forms a piperidine ring via double alkylation.
        Look for: linear diamine + dialkylating agent -> piperidine ring
        """
        # Piperidine ring pattern
        piperidine_pattern = Chem.MolFromSmarts("[#6]1-[#6]-[#7]-[#6]-[#6]-[#6]1")
        if piperidine_pattern is None:
            return False
        
        # Linear diamine patterns (potential piperidine precursors)
        diamine_patterns = [
            Chem.MolFromSmarts("N-[#6]-[#6]-[#6]-[#6]-N"),  # 1,5-diamine
            Chem.MolFromSmarts("N-[#6]-[#6]-[#6]-N"),       # 1,4-diamine that could cyclize
        ]
        diamine_patterns = [p for p in diamine_patterns if p is not None]
        
        # Alkylating agent patterns
        alkylating_patterns = [
            Chem.MolFromSmarts("[#6]-[Cl,Br,I]"),           # Alkyl halides
            Chem.MolFromSmarts("[#6]-O-S(=O)(=O)-[#6]"),    # Tosylates/mesylates
            Chem.MolFromSmarts("Cl-[#6]-[#6]-Cl"),          # Dialkyl dichloride
            Chem.MolFromSmarts("Br-[#6]-[#6]-Br"),          # Dialkyl dibromide
        ]
        alkylating_patterns = [p for p in alkylating_patterns if p is not None]
        
        # Check if products contain piperidine
        has_piperidine_product = any(
            mol.HasSubstructMatch(piperidine_pattern) for mol in products
        )
        
        if not has_piperidine_product:
            return False
        
        # Check if reactants contain diamine precursor
        has_diamine_reactant = any(
            any(mol.HasSubstructMatch(pattern) for pattern in diamine_patterns)
            for mol in reactants
        )
        
        # Check if reactants contain alkylating agent
        has_alkylating_reactant = any(
            any(mol.HasSubstructMatch(pattern) for pattern in alkylating_patterns)
            for mol in reactants
        )
        
        # Additional check: count nitrogen-containing rings formed
        reactant_piperidines = sum(
            1 for mol in reactants if mol.HasSubstructMatch(piperidine_pattern)
        )
        product_piperidines = sum(
            1 for mol in products if mol.HasSubstructMatch(piperidine_pattern)
        )
        
        ring_formed = product_piperidines > reactant_piperidines
        
        # Return True if we have evidence of piperidine formation via alkylation
        return ring_formed and (has_diamine_reactant or has_alkylating_reactant)
