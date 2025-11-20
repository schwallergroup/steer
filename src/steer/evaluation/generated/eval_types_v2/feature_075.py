"""Generated evaluation code for: SEM protecting group strategy on pyrazole"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SEMProtectionPyrazole(BaseScoring):
    """
    Evaluates routes that employ SEM (2-(Trimethylsilyl)ethoxymethyl) protecting group 
    strategy on pyrazole nitrogen to control regioselectivity during subsequent transformations.
    
    Detects SEM protection/deprotection reactions involving pyrazole substrates.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "float")
        self.target_depth = config.get("target_depth", {}).get("value", 0.5)
    
    def route_scoring(self, x) -> float:
        """Convert depth fraction to 0-10 score"""
        if x < 0:
            return 0  # Strategy not found
        
        if self.condition_type == "bool":
            return 10 if x >= 0 else 0
        else:
            # Earlier use of protection strategy is generally better
            return max(0, 10 * (1 - abs(x - self.target_depth)))
    
    def hit_condition(self, d):
        """Check if reaction involves SEM protection/deprotection of pyrazole"""
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            # Filter out None molecules
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            # Define patterns
            pyrazole_pattern = Chem.MolFromSmarts("[nH]1ncccc1")  # Pyrazole with free NH
            sem_pyrazole_pattern = Chem.MolFromSmarts("n1(COCC[Si](C)(C)C)ncccc1")  # SEM-protected pyrazole
            sem_reagent_pattern = Chem.MolFromSmarts("ClCOCC[Si](C)(C)C")  # SEM-Cl reagent
            
            if not all([pyrazole_pattern, sem_pyrazole_pattern, sem_reagent_pattern]):
                return False
            
            # Check for SEM protection reaction (pyrazole + SEM-Cl -> SEM-pyrazole)
            has_free_pyrazole_reactant = any(mol.HasSubstructMatch(pyrazole_pattern) for mol in reactants)
            has_sem_reagent = any(mol.HasSubstructMatch(sem_reagent_pattern) for mol in reactants)
            has_sem_pyrazole_product = any(mol.HasSubstructMatch(sem_pyrazole_pattern) for mol in products)
            
            protection_reaction = (has_free_pyrazole_reactant and 
                                 has_sem_reagent and 
                                 has_sem_pyrazole_product)
            
            # Check for SEM deprotection reaction (SEM-pyrazole -> pyrazole)
            has_sem_pyrazole_reactant = any(mol.HasSubstructMatch(sem_pyrazole_pattern) for mol in reactants)
            has_free_pyrazole_product = any(mol.HasSubstructMatch(pyrazole_pattern) for mol in products)
            
            deprotection_reaction = (has_sem_pyrazole_reactant and 
                                   has_free_pyrazole_product)
            
            # Check for reactions involving SEM-protected pyrazole (strategy utilization)
            involves_sem_pyrazole = (any(mol.HasSubstructMatch(sem_pyrazole_pattern) for mol in reactants) or
                                   any(mol.HasSubstructMatch(sem_pyrazole_pattern) for mol in products))
            
            return protection_reaction or deprotection_reaction or involves_sem_pyrazole
            
        except Exception:
            return False
