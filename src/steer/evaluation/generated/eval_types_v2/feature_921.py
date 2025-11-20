"""Generated evaluation code for: Protecting group strategy with THP"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class THPProtectingGroupStrategy(BaseScoring):
    """
    Evaluates synthesis routes for THP (tetrahydropyranyl) protecting group strategy.
    Checks if THP protection is used to protect alcohols during the synthesis.
    """
    
    def __init__(self, config: Dict):
        self.protecting_groups = config.get("protecting_groups", ["THP"])
        self.functional_groups_protected = config.get("functional_groups_protected", ["alcohol"])
        
        # SMARTS patterns for detection
        self.thp_pattern = Chem.MolFromSmarts("[CH2]1[CH2][CH2][CH2][CH2]O1")  # THP ring
        self.thp_ether_pattern = Chem.MolFromSmarts("CO[CH]1OCCCC1")  # THP-protected alcohol
        self.alcohol_pattern = Chem.MolFromSmarts("[CH2,CH][OH]")  # Primary/secondary alcohol
    
    def route_scoring(self, x) -> float:
        """
        Scoring function where early use of THP protection gets higher scores.
        x is the depth fraction where THP protection occurs.
        """
        if x < 0:
            return 0  # THP protection not found
        else:
            return 1 - x  # Earlier protection gets better score (closer to 1)
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves THP protection of an alcohol.
        Returns True if THP protection is detected.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            reactants_smiles = rxn_parts[0]
            products_smiles = rxn_parts[1]
            
            # Parse reactants and products
            reactant_mols = []
            for smi in reactants_smiles.split("."):
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    reactant_mols.append(mol)
            
            product_mols = []
            for smi in products_smiles.split("."):
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    product_mols.append(mol)
            
            # Check for THP protection: alcohol in reactants, THP-ether in products
            has_alcohol_reactant = any(
                mol.HasSubstructMatch(self.alcohol_pattern) 
                for mol in reactant_mols
            )
            
            has_thp_product = any(
                mol.HasSubstructMatch(self.thp_ether_pattern) 
                for mol in product_mols
            )
            
            # Also check for presence of THP reagent in reactants
            has_thp_reagent = any(
                mol.HasSubstructMatch(self.thp_pattern) 
                for mol in reactant_mols
            )
            
            # THP protection occurs if:
            # 1. Alcohol is present in reactants
            # 2. THP-protected ether is formed in products
            # 3. THP reagent is used
            return (has_alcohol_reactant and has_thp_product) or \
                   (has_thp_reagent and has_thp_product)
                   
        except Exception:
            return False
