"""Generated evaluation code for: Early stage alcohol protection with acetate"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlyAlcoholAcetateProtection(BaseScoring):
    """
    Evaluates whether alcohol protection with acetate occurs early in the synthesis route.
    Returns higher scores when acetate protection happens at earlier stages.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "continuous")
        self.target_depth = config.get("target_depth", {}).get("value", 0.2)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Protection reaction doesn't happen
        else:
            # Early protection is better - inverse relationship with depth
            if self.condition_type == "bool":
                return 1 if x <= self.target_depth else 0
            else:
                # Continuous scoring - penalize later protection
                return max(0, 1 - x) * 10
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves alcohol protection with acetate.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            reactants, products = mapped_rxn.split(">>")
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".")]
            
            # Filter out None molecules
            reactant_mols = [mol for mol in reactant_mols if mol is not None]
            product_mols = [mol for mol in product_mols if mol is not None]
            
            if not reactant_mols or not product_mols:
                return False
            
            # Check for acetate protection pattern
            return self._is_acetate_protection(reactant_mols, product_mols)
            
        except Exception:
            return False
    
    def _is_acetate_protection(self, reactants, products):
        """
        Detect if alcohol is being protected with acetate group.
        Look for: R-OH + acetylating agent -> R-O-CO-CH3
        """
        # SMARTS patterns
        alcohol_pattern = Chem.MolFromSmarts("[OH1][C,c]")  # Alcohol group
        acetate_ester_pattern = Chem.MolFromSmarts("[C,c]OC(=O)C")  # Acetate ester
        acetylating_agent_patterns = [
            Chem.MolFromSmarts("CC(=O)Cl"),  # Acetyl chloride
            Chem.MolFromSmarts("CC(=O)OC(=O)C"),  # Acetic anhydride
            Chem.MolFromSmarts("CC(=O)O")  # Acetic acid (less common)
        ]
        
        if not all([alcohol_pattern, acetate_ester_pattern] + acetylating_agent_patterns):
            return False
        
        # Check if reactants contain alcohol and acetylating agent
        has_alcohol = any(mol.HasSubstructMatch(alcohol_pattern) for mol in reactants)
        has_acetylating_agent = any(
            any(mol.HasSubstructMatch(pattern) for pattern in acetylating_agent_patterns)
            for mol in reactants
        )
        
        # Check if products contain acetate ester
        has_acetate_ester = any(mol.HasSubstructMatch(acetate_ester_pattern) for mol in products)
        
        # Additional check: ensure we're forming more acetate esters than we started with
        reactant_acetate_count = sum(
            len(mol.GetSubstructMatches(acetate_ester_pattern)) for mol in reactants
        )
        product_acetate_count = sum(
            len(mol.GetSubstructMatches(acetate_ester_pattern)) for mol in products
        )
        
        return (has_alcohol and has_acetylating_agent and 
                has_acetate_ester and product_acetate_count > reactant_acetate_count)
