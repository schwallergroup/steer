"""Generated evaluation code for: SEM protecting group strategy for imidazole"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SEMImidazoleStrategy(BaseScoring):
    """
    Evaluates synthesis routes for SEM (2-(trimethylsilyl)ethoxymethyl) protecting group 
    strategy on imidazole nitrogen. Checks for protection at specified depth and 
    deprotection at target step.
    """
    
    def __init__(self, config: Dict):
        self.protection_step = config["parameters"]["protection_step"]
        self.deprotection_step = config["parameters"]["deprotection_step"]
        
        # SMARTS patterns for SEM group and imidazole
        self.sem_pattern = Chem.MolFromSmarts("[Si](C)(C)(C)CCO[CH2]N1C=CN=C1")  # SEM-protected imidazole
        self.imidazole_pattern = Chem.MolFromSmarts("c1c[nH]cn1")  # Free imidazole NH
        self.sem_reagent_pattern = Chem.MolFromSmarts("[Si](C)(C)(C)CCO[CH2]Cl")  # SEM-Cl reagent
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Strategy not implemented
        
        # Perfect score if protection happens at target depth
        protection_score = 10 if abs(x - self.protection_step/10) < 0.05 else max(0, 8 - abs(x - self.protection_step/10) * 20)
        
        # Check if deprotection also occurs at right step
        if hasattr(self, '_deprotection_found') and self._deprotection_found:
            return min(10, protection_score + 2)
        
        return min(10, protection_score)
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction involves SEM protection of imidazole"""
        if "mapped_reaction_smiles" not in d.get("metadata", {}):
            return False
            
        rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
        rxn_parts = rxn_smiles.split(">>")
        
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0].split(".")
        products = rxn_parts[1].split(".")
        
        # Check for protection: imidazole + SEM-Cl -> SEM-protected imidazole
        has_free_imidazole = False
        has_sem_reagent = False
        has_sem_protected = False
        
        for reactant_smiles in reactants:
            try:
                mol = Chem.MolFromSmiles(reactant_smiles)
                if mol:
                    if self.imidazole_pattern.HasSubstructMatch(mol):
                        has_free_imidazole = True
                    if self.sem_reagent_pattern.HasSubstructMatch(mol):
                        has_sem_reagent = True
            except:
                continue
                
        for product_smiles in products:
            try:
                mol = Chem.MolFromSmiles(product_smiles)
                if mol and self.sem_pattern.HasSubstructMatch(mol):
                    has_sem_protected = True
            except:
                continue
        
        # Also check for deprotection: SEM-protected -> free imidazole
        if not (has_free_imidazole and has_sem_reagent and has_sem_protected):
            reactant_has_sem = any(
                Chem.MolFromSmiles(r) and self.sem_pattern.HasSubstructMatch(Chem.MolFromSmiles(r))
                for r in reactants if Chem.MolFromSmiles(r)
            )
            product_has_free = any(
                Chem.MolFromSmiles(p) and self.imidazole_pattern.HasSubstructMatch(Chem.MolFromSmiles(p))
                for p in products if Chem.MolFromSmiles(p)
            )
            
            if reactant_has_sem and product_has_free:
                self._deprotection_found = True
                return True
        
        return has_free_imidazole and has_sem_reagent and has_sem_protected
