"""Generated evaluation code for: Early stage spiro-cyclopropane assembly"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SpiroCyclopropaneAssembly(BaseScoring):
    """
    Evaluates early stage spiro-cyclopropane assembly via carbene cycloaddition.
    Detects formation of spiro-cyclopropane structures and rewards early-stage formation.
    """
    
    def __init__(self, config: Dict):
        self.timing_preference = config.get("timing", "early")
        # SMARTS pattern for spiro-cyclopropane (cyclopropane connected to another ring via spiro carbon)
        self.spiro_cyclopropane_pattern = "[C@]12([C@@]1[C,N,O][C,N,O])[C,N,O][C,N,O][C,N,O]2"
        # Carbene intermediate pattern (carbene carbon with two leaving groups)
        self.carbene_precursor_pattern = "[C](=[N+]=[N-])"  # Diazo compound pattern
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Spiro-cyclopropane formation doesn't occur
        
        if self.timing_preference == "early":
            # Early formation is better (lower depth fraction gets higher score)
            return (1 - x) * 10
        else:
            # Late formation preference
            return x * 10
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction forms a spiro-cyclopropane via carbene cycloaddition
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
                
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            # Remove None molecules (parsing failures)
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            if not reactants or not products:
                return False
            
            # Check if products contain spiro-cyclopropane that wasn't in reactants
            spiro_pattern = Chem.MolFromSmarts(self.spiro_cyclopropane_pattern)
            carbene_pattern = Chem.MolFromSmarts(self.carbene_precursor_pattern)
            
            # Count spiro-cyclopropanes in reactants and products
            reactant_spiro_count = sum(len(mol.GetSubstructMatches(spiro_pattern)) 
                                     for mol in reactants)
            product_spiro_count = sum(len(mol.GetSubstructMatches(spiro_pattern)) 
                                    for mol in products)
            
            # Check for carbene precursor in reactants (indicates carbene mechanism)
            has_carbene_precursor = any(mol.HasSubstructMatch(carbene_pattern) 
                                      for mol in reactants)
            
            # Spiro-cyclopropane formation occurred if:
            # 1. Products have more spiro-cyclopropanes than reactants
            # 2. Carbene precursor is present (indicates carbene cycloaddition mechanism)
            spiro_formed = product_spiro_count > reactant_spiro_count
            carbene_mechanism = has_carbene_precursor
            
            return spiro_formed and carbene_mechanism
            
        except Exception:
            return False
