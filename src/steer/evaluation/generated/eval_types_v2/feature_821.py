"""Generated evaluation code for: Benzophenone imine protecting group strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BenzophenoneImineProtection(BaseScoring):
    """
    Evaluates benzophenone imine protecting group strategy for amines.
    Checks for early installation of benzophenone imine protection and
    rewards routes that use this strategy early in the synthesis.
    """
    
    def __init__(self, config: Dict):
        self.timing_preference = config.get("timing", "early_installation")
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Protection strategy not found
        
        if self.timing_preference == "early_installation":
            return 1 - x  # Reward early installation (lower depth fraction)
        else:
            return x  # Reward late installation (higher depth fraction)
    
    def hit_condition(self, d) -> bool:
        """
        Check if a reaction involves benzophenone imine protection/deprotection.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        rxn_parts = mapped_rxn.split(">>")
        reactants_smiles = rxn_parts[0]
        products_smiles = rxn_parts[1]
        
        try:
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(smi) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi) for smi in products_smiles.split(".")]
            
            # Filter out None molecules
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            # Check for benzophenone imine formation (protection)
            if self._is_protection_reaction(reactants, products):
                return True
                
            # Check for benzophenone imine hydrolysis (deprotection)
            if self._is_deprotection_reaction(reactants, products):
                return True
                
        except Exception:
            return False
            
        return False
    
    def _is_protection_reaction(self, reactants, products) -> bool:
        """
        Check if reaction forms benzophenone imine from amine + benzophenone.
        """
        # Benzophenone pattern: Ph2C=O
        benzophenone_pattern = Chem.MolFromSmarts("[#6]1:[#6]:[#6]:[#6]:[#6]:[#6]:1-[#6](=[#8])-[#6]1:[#6]:[#6]:[#6]:[#6]:[#6]:1")
        
        # Primary amine pattern
        primary_amine_pattern = Chem.MolFromSmarts("[#6]-[#7H2]")
        
        # Benzophenone imine pattern: Ph2C=N-R
        benzophenone_imine_pattern = Chem.MolFromSmarts("[#6]1:[#6]:[#6]:[#6]:[#6]:[#6]:1-[#6](=[#7]-[#6])-[#6]1:[#6]:[#6]:[#6]:[#6]:[#6]:1")
        
        if not (benzophenone_pattern and primary_amine_pattern and benzophenone_imine_pattern):
            return False
        
        # Check if reactants contain benzophenone and primary amine
        has_benzophenone = any(mol.HasSubstructMatch(benzophenone_pattern) for mol in reactants)
        has_primary_amine = any(mol.HasSubstructMatch(primary_amine_pattern) for mol in reactants)
        
        # Check if products contain benzophenone imine
        has_benzophenone_imine = any(mol.HasSubstructMatch(benzophenone_imine_pattern) for mol in products)
        
        return has_benzophenone and has_primary_amine and has_benzophenone_imine
    
    def _is_deprotection_reaction(self, reactants, products) -> bool:
        """
        Check if reaction hydrolyzes benzophenone imine to release amine.
        """
        # Benzophenone imine pattern: Ph2C=N-R
        benzophenone_imine_pattern = Chem.MolFromSmarts("[#6]1:[#6]:[#6]:[#6]:[#6]:[#6]:1-[#6](=[#7]-[#6])-[#6]1:[#6]:[#6]:[#6]:[#6]:[#6]:1")
        
        # Primary amine pattern
        primary_amine_pattern = Chem.MolFromSmarts("[#6]-[#7H2]")
        
        # Benzophenone pattern: Ph2C=O
        benzophenone_pattern = Chem.MolFromSmarts("[#6]1:[#6]:[#6]:[#6]:[#6]:[#6]:1-[#6](=[#8])-[#6]1:[#6]:[#6]:[#6]:[#6]:[#6]:1")
        
        if not (benzophenone_imine_pattern and primary_amine_pattern and benzophenone_pattern):
            return False
        
        # Check if reactants contain benzophenone imine
        has_benzophenone_imine = any(mol.HasSubstructMatch(benzophenone_imine_pattern) for mol in reactants)
        
        # Check if products contain primary amine and benzophenone
        has_primary_amine = any(mol.HasSubstructMatch(primary_amine_pattern) for mol in products)
        has_benzophenone = any(mol.HasSubstructMatch(benzophenone_pattern) for mol in products)
        
        return has_benzophenone_imine and has_primary_amine and has_benzophenone
