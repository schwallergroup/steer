"""Generated evaluation code for: Benzyl protecting group for amine"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BenzylAmineProtection(BaseScoring):
    """
    Evaluates synthesis routes for the use of benzyl protecting groups on amines.
    Detects N-benzylation reactions and their corresponding hydrogenolysis deprotection.
    Rewards routes that use this protecting group strategy effectively.
    """
    
    def __init__(self, config: Dict):
        self.strategy_type = config.get("strategy_type", "protection")  # "protection" or "deprotection"
        self.require_deprotection = config.get("require_deprotection", True)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Strategy not found
        else:
            # Earlier use of protecting group strategy is better
            return 1 - x
            
    def hit_condition(self, d) -> bool:
        """Check if this reaction involves benzyl amine protection/deprotection"""
        rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
        if not rxn_smiles or ">>" not in rxn_smiles:
            return False
            
        reactants, products = rxn_smiles.split(">>")
        reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".") if r.strip()]
        product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".") if p.strip()]
        
        if not all(reactant_mols) or not all(product_mols):
            return False
            
        if self.strategy_type == "protection":
            return self._detect_benzylation(reactant_mols, product_mols)
        else:
            return self._detect_debenzylation(reactant_mols, product_mols)
    
    def _detect_benzylation(self, reactants, products) -> bool:
        """Detect N-benzylation reaction (protection step)"""
        # Pattern for primary or secondary amine
        amine_pattern = Chem.MolFromSmarts("[NH2,NH1]")
        # Pattern for benzyl halide or benzyl alcohol derivatives
        benzyl_electrophile = Chem.MolFromSmarts("c1ccccc1C[Cl,Br,I,OH]")
        # Pattern for N-benzyl amine product
        nbenzyl_pattern = Chem.MolFromSmarts("c1ccccc1CN")
        
        if not all([amine_pattern, benzyl_electrophile, nbenzyl_pattern]):
            return False
            
        # Check if reactants contain amine and benzyl electrophile
        has_amine = any(mol.HasSubstructMatch(amine_pattern) for mol in reactants)
        has_benzyl_electrophile = any(mol.HasSubstructMatch(benzyl_electrophile) for mol in reactants)
        
        # Check if product contains N-benzyl group
        has_nbenzyl_product = any(mol.HasSubstructMatch(nbenzyl_pattern) for mol in products)
        
        return has_amine and has_benzyl_electrophile and has_nbenzyl_product
    
    def _detect_debenzylation(self, reactants, products) -> bool:
        """Detect hydrogenolytic debenzylation (deprotection step)"""
        # Pattern for N-benzyl amine
        nbenzyl_pattern = Chem.MolFromSmarts("c1ccccc1CN")
        # Pattern for free amine
        amine_pattern = Chem.MolFromSmarts("[NH2,NH1]")
        # Pattern for toluene (byproduct of hydrogenolysis)
        toluene_pattern = Chem.MolFromSmarts("c1ccccc1C")
        
        if not all([nbenzyl_pattern, amine_pattern, toluene_pattern]):
            return False
            
        # Check if reactant contains N-benzyl group
        has_nbenzyl_reactant = any(mol.HasSubstructMatch(nbenzyl_pattern) for mol in reactants)
        
        # Check if products contain free amine and toluene
        has_amine_product = any(mol.HasSubstructMatch(amine_pattern) for mol in products)
        has_toluene = any(mol.HasSubstructMatch(toluene_pattern) for mol in products)
        
        return has_nbenzyl_reactant and has_amine_product and has_toluene
