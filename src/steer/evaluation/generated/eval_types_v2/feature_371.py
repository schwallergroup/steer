"""Generated evaluation code for: Tandem deprotection-cyclization for lactam formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class TandemDeprotectionCyclization(BaseScoring):
    """
    Evaluates synthesis routes for the presence of tandem deprotection-cyclization
    reactions that form lactam rings, specifically targeting Boc deprotection
    followed by intramolecular cyclization to form 6-membered lactam rings.
    """
    
    def __init__(self, config: Dict):
        self.target_ring_size = config.get("parameters", {}).get("ring_size", 6)
        self.condition_type = config.get("target_depth", {}).get("type", "bool")
        self.target_depth = config.get("target_depth", {}).get("value", -1)

    def route_scoring(self, x) -> float:
        if self.condition_type == "bool":
            if self.target_depth == -1:  # Positive if condition not met
                return 1 if x < 0 else 0
        else:
            if x < 0:
                return 0
            return abs(x - self.target_depth)

    def hit_condition(self, d):
        """
        Detects tandem deprotection-cyclization by checking for:
        1. Boc group removal (loss of tert-butoxycarbonyl)
        2. Simultaneous lactam ring formation
        3. Ring size matching target (default 6-membered)
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
                
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            reactants = [Chem.MolFromSmiles(smi) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi) for smi in products_smiles.split(".")]
            
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            if not reactants or not products:
                return False
            
            # Check for Boc deprotection
            has_boc_deprotection = self._detect_boc_deprotection(reactants, products)
            
            # Check for lactam ring formation
            has_lactam_formation = self._detect_lactam_formation(reactants, products)
            
            return has_boc_deprotection and has_lactam_formation
            
        except Exception:
            return False

    def _detect_boc_deprotection(self, reactants, products):
        """Detect removal of Boc protecting group"""
        # Boc group pattern: tert-butoxycarbonyl
        boc_pattern = Chem.MolFromSmarts("[NX3][C](=[O])[O][C]([CH3])([CH3])[CH3]")
        if boc_pattern is None:
            return False
            
        # Check if any reactant contains Boc group
        has_boc_reactant = any(mol.HasSubstructMatch(boc_pattern) for mol in reactants)
        
        # Check if products lack the Boc group (or have fewer)
        reactant_boc_count = sum(len(mol.GetSubstructMatches(boc_pattern)) for mol in reactants)
        product_boc_count = sum(len(mol.GetSubstructMatches(boc_pattern)) for mol in products)
        
        return has_boc_reactant and product_boc_count < reactant_boc_count

    def _detect_lactam_formation(self, reactants, products):
        """Detect formation of lactam ring with target ring size"""
        # Lactam pattern: cyclic amide
        if self.target_ring_size == 6:
            # 6-membered lactam (piperidone-like)
            lactam_pattern = Chem.MolFromSmarts("[NX3]1[C]([#6])[#6][#6][#6][C]1=[O]")
        elif self.target_ring_size == 5:
            # 5-membered lactam (pyrrolidinone-like)
            lactam_pattern = Chem.MolFromSmarts("[NX3]1[C]([#6])[#6][#6][C]1=[O]")
        elif self.target_ring_size == 7:
            # 7-membered lactam
            lactam_pattern = Chem.MolFromSmarts("[NX3]1[C]([#6])[#6][#6][#6][#6][C]1=[O]")
        else:
            # General lactam pattern for other ring sizes
            lactam_pattern = Chem.MolFromSmarts("[NX3]1[C](=[O])[#6]1")
            
        if lactam_pattern is None:
            return False
        
        # Count lactam rings in reactants vs products
        reactant_lactam_count = sum(len(mol.GetSubstructMatches(lactam_pattern)) for mol in reactants)
        product_lactam_count = sum(len(mol.GetSubstructMatches(lactam_pattern)) for mol in products)
        
        # Ring formation means more lactams in products than reactants
        return product_lactam_count > reactant_lactam_count
