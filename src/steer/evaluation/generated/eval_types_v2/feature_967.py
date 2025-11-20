"""Generated evaluation code for: Phthalimide protecting group strategy for amine"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class PhthalimideProtectingGroup(BaseScoring):
    """
    Evaluates the use of phthalimide protecting group strategy for primary amines.
    Checks if phthaloyl chloride is used to install phthalimide protection on primary amines.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "bool")
        self.target_depth = config.get("target_depth", {}).get("value", 0)
        
        # SMARTS patterns for detection
        self.phthalimide_pattern = "C1C(=O)c2ccccc2C(=O)N1"  # Phthalimide core
        self.phthaloyl_chloride_pattern = "C(=O)Clc1ccccc1C(=O)Cl"  # Phthaloyl chloride
        self.primary_amine_pattern = "[NH2]"  # Primary amine
        
    def route_scoring(self, x) -> float:
        """Convert depth fraction to score (0-10)"""
        if self.condition_type == "bool":
            if self.target_depth == -1:  # Positive if condition not met
                return 1 if x < 0 else 0
        else:
            if x < 0:
                return 0  # Protection strategy not used
            # Early installation of protecting group is generally better
            return max(0, 1 - x)
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction involves phthalimide protection of primary amine"""
        try:
            mapped_rxn = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not mapped_rxn or ">>" not in mapped_rxn:
                return False
                
            products_smiles, reactants_smiles = mapped_rxn.split(">>")
            
            # Parse molecules
            products = [Chem.MolFromSmiles(smi) for smi in products_smiles.split(".")]
            reactants = [Chem.MolFromSmiles(smi) for smi in reactants_smiles.split(".")]
            
            # Remove None molecules (parsing failures)
            products = [mol for mol in products if mol is not None]
            reactants = [mol for mol in reactants if mol is not None]
            
            if not products or not reactants:
                return False
            
            # Create SMARTS patterns
            phthalimide_smarts = Chem.MolFromSmarts(self.phthalimide_pattern)
            phthaloyl_chloride_smarts = Chem.MolFromSmarts(self.phthaloyl_chloride_pattern)
            primary_amine_smarts = Chem.MolFromSmarts(self.primary_amine_pattern)
            
            if None in [phthalimide_smarts, phthaloyl_chloride_smarts, primary_amine_smarts]:
                return False
            
            # Check if reactants contain phthaloyl chloride and primary amine
            has_phthaloyl_chloride = any(
                mol.HasSubstructMatch(phthaloyl_chloride_smarts) for mol in reactants
            )
            
            has_primary_amine = any(
                mol.HasSubstructMatch(primary_amine_smarts) for mol in reactants
            )
            
            # Check if products contain phthalimide
            has_phthalimide_product = any(
                mol.HasSubstructMatch(phthalimide_smarts) for mol in products
            )
            
            # Additional check: primary amine should be consumed (not in products)
            primary_amine_consumed = not any(
                mol.HasSubstructMatch(primary_amine_smarts) for mol in products
            )
            
            # Phthalimide protection reaction should have:
            # 1. Phthaloyl chloride in reactants
            # 2. Primary amine in reactants  
            # 3. Phthalimide group in products
            # 4. Primary amine consumed (not in products)
            return (has_phthaloyl_chloride and 
                   has_primary_amine and 
                   has_phthalimide_product and 
                   primary_amine_consumed)
                   
        except Exception:
            return False
