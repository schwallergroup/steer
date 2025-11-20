"""Generated evaluation code for: Corey-Chaykovsky cyclopropanation for ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class CoreyChaykovsky(BaseScoring):
    """
    Evaluates synthesis routes for Corey-Chaykovsky cyclopropanation reactions.
    Detects the formation of cyclopropyl rings from alpha,beta-unsaturated esters
    using sulfur ylide methodology.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "fraction")
        self.target_depth = config.get("target_depth", {}).get("value", 0.5)
    
    def route_scoring(self, x) -> float:
        """Convert depth fraction to 0-10 score"""
        if x < 0:
            return 0  # Reaction not found
        if self.condition_type == "bool":
            return 10  # Found the reaction
        else:
            # Earlier in synthesis is better for this strategic reaction
            return 10 * (1 - x)
    
    def hit_condition(self, d) -> bool:
        """Check if reaction node represents Corey-Chaykovsky cyclopropanation"""
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            
            if not all(reactants) or not all(products):
                return False
            
            # Check for cyclopropyl ring formation
            cyclopropyl_formed = self._cyclopropyl_ring_formed(reactants, products)
            
            # Check for alpha,beta-unsaturated ester substrate
            has_unsaturated_ester = any(self._has_alpha_beta_unsaturated_ester(mol) for mol in reactants)
            
            # Check for sulfur ylide reagent (sulfoxonium or sulfonium ylide)
            has_sulfur_ylide = any(self._has_sulfur_ylide(mol) for mol in reactants)
            
            return cyclopropyl_formed and has_unsaturated_ester and has_sulfur_ylide
            
        except Exception:
            return False
    
    def _cyclopropyl_ring_formed(self, reactants, products) -> bool:
        """Check if cyclopropyl ring is formed in the reaction"""
        # Count cyclopropyl rings in reactants vs products
        cyclopropyl_pattern = Chem.MolFromSmarts("[#6]1[#6][#6]1")
        
        reactant_cyclopropyl = sum(len(mol.GetSubstructMatches(cyclopropyl_pattern)) 
                                 for mol in reactants)
        product_cyclopropyl = sum(len(mol.GetSubstructMatches(cyclopropyl_pattern)) 
                                for mol in products)
        
        return product_cyclopropyl > reactant_cyclopropyl
    
    def _has_alpha_beta_unsaturated_ester(self, mol) -> bool:
        """Check for alpha,beta-unsaturated ester pattern"""
        # Pattern for alpha,beta-unsaturated ester: C=C-C(=O)-O
        unsaturated_ester_pattern = Chem.MolFromSmarts("[#6]=[#6]-[#6](=[#8])-[#8]")
        return mol.HasSubstructMatch(unsaturated_ester_pattern)
    
    def _has_sulfur_ylide(self, mol) -> bool:
        """Check for sulfur ylide reagent (sulfoxonium or sulfonium ylide)"""
        # Sulfoxonium ylide pattern: [S+](=O)-[C-] or similar
        sulfoxonium_pattern = Chem.MolFromSmarts("[#16+]([#8])[#6-]")
        # Sulfonium ylide pattern: [S+]-[C-]
        sulfonium_pattern = Chem.MolFromSmarts("[#16+][#6-]")
        # Dimethyl sulfoxonium methylide-like patterns
        dmso_ylide_pattern = Chem.MolFromSmarts("[#16](=[#8])([#6])[#6][#6-]")
        
        return (mol.HasSubstructMatch(sulfoxonium_pattern) or 
                mol.HasSubstructMatch(sulfonium_pattern) or
                mol.HasSubstructMatch(dmso_ylide_pattern))
