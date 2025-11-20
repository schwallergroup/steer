"""Generated evaluation code for: Stepwise benzyl protecting group removal"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class StepwiseBenzylDeprotection(MultiRxnCondBase):
    """
    Evaluates routes for stepwise removal of benzyl protecting groups via hydrogenolysis.
    Checks if the route contains stepwise deprotection of O-benzyl and N-benzyl groups
    using the same removal method (hydrogenolysis).
    """
    
    def __init__(self, config):
        self.group_count = config.get("group_count", 2)
        self.removal_method = config.get("removal_method", "hydrogenolysis")
        self.selectivity = config.get("selectivity", "stepwise")
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Find all benzyl deprotection reactions
        deprotection_reactions = []
        for i, rxn in enumerate(reactions):
            if self.is_benzyl_deprotection(rxn):
                deprotection_reactions.append((i, rxn))
        
        # Check if we have stepwise removal (at least 2 separate reactions)
        if len(deprotection_reactions) < self.group_count:
            return False, len(reactions)
            
        # Verify reactions are separated (not in same step)
        reaction_indices = [idx for idx, _ in deprotection_reactions]
        if len(set(reaction_indices)) < self.group_count:
            return False, len(reactions)
            
        # Check for both O-benzyl and N-benzyl removals
        has_o_benzyl = any(self.has_o_benzyl_removal(rxn) for _, rxn in deprotection_reactions)
        has_n_benzyl = any(self.has_n_benzyl_removal(rxn) for _, rxn in deprotection_reactions)
        
        condition = has_o_benzyl and has_n_benzyl
        return condition, len(reactions)
    
    def is_benzyl_deprotection(self, rxn):
        """Check if reaction involves benzyl group removal via hydrogenolysis"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        # Check for hydrogen gas (H2) in reactants - indicator of hydrogenolysis
        if "[H][H]" not in reactants and "H" not in reactants:
            return False
            
        # Check for toluene or benzyl alcohol formation (common byproducts)
        product_mols = products.split(".")
        has_benzyl_byproduct = any("c1ccccc1" in prod for prod in product_mols)
        
        return has_benzyl_byproduct
    
    def has_o_benzyl_removal(self, rxn):
        """Check if reaction removes O-benzyl protecting group"""
        o_benzyl_pattern = "[O]Cc1ccccc1"  # O-benzyl pattern
        return self.detect_protecting_group_removal(rxn, o_benzyl_pattern)
    
    def has_n_benzyl_removal(self, rxn):
        """Check if reaction removes N-benzyl protecting group"""
        n_benzyl_pattern = "[N]Cc1ccccc1"  # N-benzyl pattern  
        return self.detect_protecting_group_removal(rxn, n_benzyl_pattern)
    
    def detect_protecting_group_removal(self, rxn, pattern):
        """Generic method to detect protecting group removal"""
        try:
            rxn_parts = rxn.split(">>")
            reactants_smiles = rxn_parts[0].split(".")[0]  # Main substrate
            products_smiles = rxn_parts[1].split(".")[0]   # Main product
            
            reactant_mol = Chem.MolFromSmiles(reactants_smiles)
            product_mol = Chem.MolFromSmiles(products_smiles)
            
            if reactant_mol is None or product_mol is None:
                return False
                
            pattern_mol = Chem.MolFromSmarts(pattern)
            if pattern_mol is None:
                return False
            
            # Pattern present in reactant but not in product = removal
            reactant_has_pattern = reactant_mol.HasSubstructMatch(pattern_mol)
            product_has_pattern = product_mol.HasSubstructMatch(pattern_mol)
            
            return reactant_has_pattern and not product_has_pattern
            
        except Exception:
            return False
