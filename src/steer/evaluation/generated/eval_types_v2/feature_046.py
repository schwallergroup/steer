"""Generated evaluation code for: Krapcho decarboxylation for ester removal"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class KrapchodecarboxylationDepth(BaseScoring):
    """
    Evaluates synthesis routes for the presence of Krapcho decarboxylation reactions.
    Krapcho decarboxylation removes ethoxycarbonyl groups (COOEt) that are alpha to nitriles
    or other electron-withdrawing groups under basic conditions with DMSO.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "distance")
        self.target_depth = config.get("target_depth", {}).get("value", 0.5)
    
    def route_scoring(self, x) -> float:
        if self.condition_type == "bool":
            if self.target_depth == -1:  # Positive if condition not met
                return 1 if x < 0 else 0
            else:
                return 1 if x >= 0 else 0
        else:
            if x < 0:
                return 0  # Reaction doesn't occur
            return max(0, 1 - abs(x - self.target_depth))
    
    def hit_condition(self, d):
        """
        Detects Krapcho decarboxylation by identifying the loss of ethoxycarbonyl
        group alpha to electron-withdrawing groups like nitriles.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            reactants_smiles, products_smiles = mapped_rxn.split(">>")
            
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            if not all(reactants) or not all(products):
                return False
                
            # Look for substrate pattern: ethyl ester alpha to nitrile
            # Pattern: C(=O)OCC adjacent to C#N or other EWG
            substrate_pattern = Chem.MolFromSmarts("[C:1](=[O:2])[O:3][CH2:4][CH3:5].[C:6][C:7]#[N:8]")
            krapcho_substrate = Chem.MolFromSmarts("C(=O)OCC")  # Ethoxycarbonyl group
            nitrile_adjacent = Chem.MolFromSmarts("CC#N")  # Nitrile adjacent carbon
            
            # Check if any reactant contains the substrate pattern
            has_substrate = False
            substrate_mol = None
            
            for reactant in reactants:
                if reactant.HasSubstructMatch(krapcho_substrate):
                    # Check if ethoxycarbonyl is adjacent to nitrile or EWG
                    if (reactant.HasSubstructMatch(nitrile_adjacent) or 
                        reactant.HasSubstructMatch(Chem.MolFromSmarts("C(C#N)C(=O)OCC")) or
                        reactant.HasSubstructMatch(Chem.MolFromSmarts("C(C(=O)OCC)C#N"))):
                        has_substrate = True
                        substrate_mol = reactant
                        break
            
            if not has_substrate:
                return False
                
            # Check if product has lost the ethoxycarbonyl group
            # The substrate should be decarboxylated (COOEt removed, replaced by H)
            ethoxycarbonyl_lost = True
            for product in products:
                if product.HasSubstructMatch(krapcho_substrate):
                    # Still contains ethoxycarbonyl - not a Krapcho reaction
                    ethoxycarbonyl_lost = False
                    break
                    
            # Additional check: should produce ethanol or CO2 as byproduct
            byproducts = ["CCO", "O=C=O", "C(=O)O"]  # EtOH, CO2, or formic acid
            has_expected_byproduct = any(
                any(Chem.MolFromSmiles(bp).HasSubstructMatch(product) or 
                    product.HasSubstructMatch(Chem.MolFromSmiles(bp)) for product in products)
                for bp in byproducts
            )
            
            return ethoxycarbonyl_lost and (has_expected_byproduct or len(products) > 1)
            
        except Exception:
            return False
