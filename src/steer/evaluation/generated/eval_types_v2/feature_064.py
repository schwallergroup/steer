"""Generated evaluation code for: Early acetonide protection of amino diol"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlyAcetonideProtection(BaseScoring):
    """
    Checks if acetonide protection of amino diol occurs early in the synthesis route.
    Acetonide should simultaneously protect both diol and amine functionalities.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "bool")
        self.target_depth = config.get("target_depth", {}).get("value", 1)
        
        # SMARTS patterns for acetonide-protected amino diol
        self.acetonide_aminodiol_pattern = "[N,NH,NH2]C[C,CH]1OC(C)(C)O[C,CH]1"
        
        # Patterns for unprotected substrates
        self.diol_pattern = "[CH,CH2][OH][CH,CH2][OH]"
        self.aminodiol_pattern = "[N,NH,NH2][CH,CH2][CH,CH2][OH][CH,CH2][OH]"
    
    def route_scoring(self, x) -> float:
        if self.condition_type == "bool":
            if self.target_depth == -1:  # Positive if condition not met
                return 1 if x < 0 else 0
        else:
            if x < 0:
                return 0  # Protection doesn't happen
            # Early protection is better (lower depth fraction)
            if x <= 0.2:  # Very early
                return 1.0
            elif x <= 0.4:  # Early
                return 0.8
            else:  # Late protection
                return 0.3
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves acetonide protection of amino diol
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            reactants_smiles, products_smiles = mapped_rxn.split(">>")
            
            # Parse reactants and products
            reactant_mols = []
            for smi in reactants_smiles.split("."):
                mol = Chem.MolFromSmiles(smi.strip())
                if mol is not None:
                    reactant_mols.append(mol)
            
            product_mols = []
            for smi in products_smiles.split("."):
                mol = Chem.MolFromSmiles(smi.strip())
                if mol is not None:
                    product_mols.append(mol)
            
            # Check if we have acetonide formation
            has_acetonide_product = any(
                mol.HasSubstructMatch(Chem.MolFromSmarts(self.acetonide_aminodiol_pattern))
                for mol in product_mols
            )
            
            if not has_acetonide_product:
                return False
            
            # Check if reactants contain unprotected amino diol
            has_aminodiol_reactant = any(
                mol.HasSubstructMatch(Chem.MolFromSmarts(self.aminodiol_pattern)) or
                (mol.HasSubstructMatch(Chem.MolFromSmarts(self.diol_pattern)) and 
                 mol.HasSubstructMatch(Chem.MolFromSmarts("[N,NH,NH2]")))
                for mol in reactant_mols
            )
            
            # Check for acetonide precursor (acetone or 2,2-dimethoxypropane)
            acetonide_reagents = [
                "CC(C)=O",  # acetone
                "CC(C)(OC)OC",  # 2,2-dimethoxypropane
                "CC(OC)(OC)C"   # alternative DMP pattern
            ]
            
            has_acetonide_reagent = any(
                any(mol.HasSubstructMatch(Chem.MolFromSmiles(reagent)) 
                    for mol in reactant_mols)
                for reagent in acetonide_reagents
            )
            
            return has_aminodiol_reactant and has_acetonide_reagent and has_acetonide_product
            
        except Exception:
            return False
