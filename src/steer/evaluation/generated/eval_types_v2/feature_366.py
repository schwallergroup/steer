"""Generated evaluation code for: Tandem Boc deprotection lactam cyclization"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class TandemBocDeprotectionLactamCyclization(BaseScoring):
    """
    Evaluates routes for tandem Boc deprotection followed by lactam cyclization.
    This reaction involves removal of a Boc protecting group and subsequent 
    intramolecular cyclization to form a lactam ring in a single operation.
    """
    
    def __init__(self, config: Dict):
        self.pot_economy = config.get("pot_economy", True)
        self.transformation_count = config.get("transformation_count", 2)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Reaction doesn't happen
        else:
            return 1 - x  # Earlier occurrence is better for synthetic efficiency
    
    def hit_condition(self, d) -> bool:
        """
        Check if a reaction node represents tandem Boc deprotection + lactam cyclization
        """
        rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
        if not rxn_smiles or ">>" not in rxn_smiles:
            return False
            
        reactants_smiles, products_smiles = rxn_smiles.split(">>")
        
        try:
            # Parse reactants and products
            reactant_mols = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            product_mols = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            if not all(reactant_mols) or not all(product_mols):
                return False
            
            # Check for Boc deprotection
            boc_removed = self._detect_boc_deprotection(reactant_mols, product_mols)
            
            # Check for lactam formation
            lactam_formed = self._detect_lactam_formation(reactant_mols, product_mols)
            
            return boc_removed and lactam_formed
            
        except Exception:
            return False
    
    def _detect_boc_deprotection(self, reactants, products):
        """Detect removal of Boc protecting group (tert-butoxycarbonyl)"""
        # Boc group pattern: -NHC(=O)OC(C)(C)C or variations
        boc_patterns = [
            "NC(=O)OC(C)(C)C",  # Linear Boc
            "[NH]C(=O)OC(C)(C)C",  # Boc on secondary amine
        ]
        
        # Check if any reactant contains Boc group
        boc_in_reactants = False
        for reactant in reactants:
            for pattern in boc_patterns:
                boc_smarts = Chem.MolFromSmarts(pattern)
                if boc_smarts and reactant.HasSubstructMatch(boc_smarts):
                    boc_in_reactants = True
                    break
        
        # Check if products lack the Boc group (indicating deprotection)
        boc_in_products = False
        for product in products:
            for pattern in boc_patterns:
                boc_smarts = Chem.MolFromSmarts(pattern)
                if boc_smarts and product.HasSubstructMatch(boc_smarts):
                    boc_in_products = True
                    break
        
        return boc_in_reactants and not boc_in_products
    
    def _detect_lactam_formation(self, reactants, products):
        """Detect formation of lactam ring (cyclic amide)"""
        # Lactam patterns for different ring sizes
        lactam_patterns = [
            "C1CCC(=O)N1",    # 5-membered lactam (pyrrolidin-2-one)
            "C1CCCC(=O)N1",   # 6-membered lactam (piperidin-2-one)
            "C1CCCCC(=O)N1",  # 7-membered lactam
            "C1CC(=O)N1",     # 4-membered lactam (beta-lactam)
        ]
        
        # Count lactam rings in reactants
        lactam_count_reactants = 0
        for reactant in reactants:
            for pattern in lactam_patterns:
                lactam_smarts = Chem.MolFromSmarts(pattern)
                if lactam_smarts:
                    lactam_count_reactants += len(reactant.GetSubstructMatches(lactam_smarts))
        
        # Count lactam rings in products
        lactam_count_products = 0
        for product in products:
            for pattern in lactam_patterns:
                lactam_smarts = Chem.MolFromSmarts(pattern)
                if lactam_smarts:
                    lactam_count_products += len(product.GetSubstructMatches(lactam_smarts))
        
        # Lactam formation means more lactam rings in products than reactants
        return lactam_count_products > lactam_count_reactants
