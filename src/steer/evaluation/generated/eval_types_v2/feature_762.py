"""Generated evaluation code for: SEM protecting group for heterocycle nitrogen"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SEMProtectionStrategy(BaseScoring):
    """
    Evaluates synthesis routes for the use of SEM (2-(trimethylsilyl)ethoxymethyl) 
    protecting group on heterocycle nitrogen atoms. SEM protection prevents 
    interference of acidic N-H protons during subsequent reactions like SNAr.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "fraction")
        self.target_depth = config.get("target_depth", {}).get("value", 0.0)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # SEM protection not used
        else:
            # Earlier protection is generally better
            return max(0, 10 * (1 - x))
    
    def hit_condition(self, d) -> bool:
        """
        Check if a reaction involves SEM protection of a heterocycle nitrogen.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            reactants_smiles, products_smiles = mapped_rxn.split(">>")
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            # Filter out None molecules
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            return self._detect_sem_protection(reactants, products)
            
        except:
            return False
    
    def _detect_sem_protection(self, reactants, products) -> bool:
        """
        Detect SEM protection by looking for:
        1. Heterocycle with NH in reactants
        2. Same heterocycle with N-SEM in products
        3. Presence of SEM reagent in reactants
        """
        # SEM group pattern: N-CH2-O-CH2-CH2-Si(CH3)3
        sem_pattern = Chem.MolFromSmarts("[N]-[CH2]-[O]-[CH2]-[CH2]-[Si]([CH3])([CH3])[CH3]")
        
        # Common heterocycle NH patterns
        heterocycle_nh_patterns = [
            Chem.MolFromSmarts("[nH]1cccc1"),  # pyrrole NH
            Chem.MolFromSmarts("[nH]1ccccc1"), # indole NH
            Chem.MolFromSmarts("[nH]1ccnc1"),  # imidazole NH
            Chem.MolFromSmarts("[nH]1cnnc1"),  # pyrazole NH
            Chem.MolFromSmarts("[nH]1cncc1"),  # pyrazine NH
        ]
        
        # SEM reagent patterns (SEM-Cl is most common)
        sem_reagent_patterns = [
            Chem.MolFromSmarts("Cl-[CH2]-[O]-[CH2]-[CH2]-[Si]([CH3])([CH3])[CH3]"),  # SEM-Cl
            Chem.MolFromSmarts("Br-[CH2]-[O]-[CH2]-[CH2]-[Si]([CH3])([CH3])[CH3]"),  # SEM-Br
        ]
        
        # Check if products contain SEM-protected heterocycle
        has_sem_product = False
        for product in products:
            if product.HasSubstructMatch(sem_pattern):
                has_sem_product = True
                break
        
        if not has_sem_product:
            return False
        
        # Check if reactants contain heterocycle NH
        has_heterocycle_nh = False
        for reactant in reactants:
            for pattern in heterocycle_nh_patterns:
                if reactant.HasSubstructMatch(pattern):
                    has_heterocycle_nh = True
                    break
            if has_heterocycle_nh:
                break
        
        # Check if reactants contain SEM reagent
        has_sem_reagent = False
        for reactant in reactants:
            for pattern in sem_reagent_patterns:
                if reactant.HasSubstructMatch(pattern):
                    has_sem_reagent = True
                    break
            if has_sem_reagent:
                break
        
        return has_sem_product and (has_heterocycle_nh or has_sem_reagent)
