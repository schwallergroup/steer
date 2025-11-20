"""Generated evaluation code for: Combined dehydration and Boc deprotection step"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BocDehydrationStep(BaseScoring):
    """
    Evaluates routes that perform combined Boc deprotection and dehydration in a single step.
    Detects reactions where tert-butoxycarbonyl (Boc) protecting group removal occurs
    simultaneously with alkene formation via acid-promoted dehydration.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "float")
        self.target_depth = config.get("target_depth", {}).get("value", 0.5)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Combined step doesn't occur
        else:
            if self.condition_type == "bool":
                return 10  # Reward finding the combined step
            else:
                # Earlier in route is better for efficiency
                return max(0, 10 * (1 - abs(x - self.target_depth)))
    
    def hit_condition(self, d) -> bool:
        """Check if reaction performs both Boc deprotection and dehydration"""
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
                
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            reactants = [Chem.MolFromSmiles(smi) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi) for smi in products_smiles.split(".")]
            
            if not all(reactants) or not all(products):
                return False
            
            # Check for Boc deprotection
            boc_removed = self._detect_boc_removal(reactants, products)
            
            # Check for dehydration (alkene formation)
            dehydration_occurred = self._detect_dehydration(reactants, products)
            
            # Check for acid reagent
            acid_present = self._detect_acid_reagent(reactants)
            
            return boc_removed and dehydration_occurred and acid_present
            
        except Exception:
            return False
    
    def _detect_boc_removal(self, reactants, products) -> bool:
        """Detect if Boc protecting group is removed"""
        # Boc pattern: tert-butoxycarbonyl
        boc_pattern = Chem.MolFromSmarts("[NX3][C](=O)OC(C)(C)C")
        if not boc_pattern:
            return False
            
        # Check if any reactant has Boc group
        boc_in_reactants = any(mol.HasSubstructMatch(boc_pattern) for mol in reactants)
        
        # Check if products lack the Boc group (or have fewer)
        boc_in_products = any(mol.HasSubstructMatch(boc_pattern) for mol in products)
        
        return boc_in_reactants and not boc_in_products
    
    def _detect_dehydration(self, reactants, products) -> bool:
        """Detect alkene formation via dehydration"""
        # Count C=C double bonds
        alkene_pattern = Chem.MolFromSmarts("C=C")
        if not alkene_pattern:
            return False
            
        reactant_alkenes = sum(len(mol.GetSubstructMatches(alkene_pattern)) for mol in reactants)
        product_alkenes = sum(len(mol.GetSubstructMatches(alkene_pattern)) for mol in products)
        
        # Also check for water formation
        water_formed = any(Chem.MolToSmiles(mol) == "O" for mol in products)
        
        return product_alkenes > reactant_alkenes and water_formed
    
    def _detect_acid_reagent(self, reactants) -> bool:
        """Detect presence of acid reagent"""
        common_acids = [
            "O=S(=O)(O)O",  # H2SO4
            "Cl",           # HCl
            "[H+]",         # Generic acid
            "CC(=O)O",      # Acetic acid
            "O=S(=O)(O)C(F)(F)F",  # Triflic acid
        ]
        
        reactant_smiles = [Chem.MolToSmiles(mol) for mol in reactants]
        
        return any(acid in reactant_smiles for acid in common_acids)
