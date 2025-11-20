"""Generated evaluation code for: Late stage Wittig olefination"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageWittig(BaseScoring):
    """
    Evaluates synthesis routes for late-stage Wittig olefination reactions.
    Scores routes based on how late in the synthesis a Wittig reaction occurs,
    with preference for reactions happening within the specified depth threshold.
    """
    
    def __init__(self, config: Dict):
        self.depth_threshold = config.get("depth_threshold", 2)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # No Wittig reaction found
        
        # Convert depth fraction to score (0-10)
        # Earlier reactions (higher x values) get lower scores
        # Later reactions (lower x values) get higher scores
        if x <= self.depth_threshold / 10.0:  # Within threshold
            return 10 - (x * 50)  # Scale appropriately for late stage
        else:
            return max(0, 5 - (x * 10))  # Reduced score for earlier reactions
    
    def hit_condition(self, d) -> bool:
        """
        Detects Wittig olefination reactions by looking for:
        1. Formation of C=C double bond
        2. Presence of phosphonium ylide or triphenylphosphine oxide as byproduct
        3. Characteristic reaction pattern
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
        
        try:
            reactants_smiles, products_smiles = mapped_rxn.split(">>")
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            
            # Filter out None molecules
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            if not reactants or not products:
                return False
            
            # Check for triphenylphosphine oxide as product (common Wittig byproduct)
            ppo_pattern = Chem.MolFromSmarts("[P](=O)(c1ccccc1)(c2ccccc2)c3ccccc3")
            has_ppo = any(mol.HasSubstructMatch(ppo_pattern) for mol in products if mol)
            
            # Check for phosphonium ylide in reactants
            ylide_pattern = Chem.MolFromSmarts("[P+](c1ccccc1)(c2ccccc2)(c3ccccc3)[C-]")
            has_ylide = any(mol.HasSubstructMatch(ylide_pattern) for mol in reactants if mol)
            
            # Alternative: check for aldehyde/ketone + phosphonium salt pattern
            carbonyl_pattern = Chem.MolFromSmarts("[C]=[O]")
            phosphonium_pattern = Chem.MolFromSmarts("[P+](c1ccccc1)(c2ccccc2)c3ccccc3")
            
            has_carbonyl = any(mol.HasSubstructMatch(carbonyl_pattern) for mol in reactants if mol)
            has_phosphonium = any(mol.HasSubstructMatch(phosphonium_pattern) for mol in reactants if mol)
            
            # Check for alkene formation in products
            alkene_pattern = Chem.MolFromSmarts("C=C")
            has_alkene_product = any(mol.HasSubstructMatch(alkene_pattern) for mol in products if mol)
            
            # Wittig reaction indicators:
            # 1. Ylide + carbonyl -> alkene + PPh3O, or
            # 2. Phosphonium salt + carbonyl -> alkene + PPh3O (with base)
            wittig_condition1 = has_ylide and has_alkene_product and has_ppo
            wittig_condition2 = has_phosphonium and has_carbonyl and has_alkene_product and has_ppo
            
            return wittig_condition1 or wittig_condition2
            
        except Exception:
            return False
