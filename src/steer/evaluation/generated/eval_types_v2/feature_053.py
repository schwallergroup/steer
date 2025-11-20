"""Generated evaluation code for: Benzyl ether protecting tertiary alcohol with alkyne"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BenzylEtherAlkyneIncompatibility(MultiRxnCondBase):
    """
    Evaluates routes for incompatible use of benzyl ether protection on tertiary alcohols
    when terminal alkynes or alkenes are present in the molecule.
    
    Benzyl ether deprotection typically uses hydrogenation conditions (Pd/C, H2) 
    which would also reduce terminal alkynes and alkenes, making this protection
    strategy incompatible.
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.allow_incompatible = config.get("allow_incompatible", False)
        
        # SMARTS patterns for detection
        self.benzyl_ether_pattern = "[CH2]c1ccccc1-O-[CH]([CH3,CH2])([CH3,CH2])"  # Benzyl ether on tertiary carbon
        self.terminal_alkyne_pattern = "[CH]#[CH]"  # Terminal alkyne
        self.alkene_pattern = "[CH]=[CH]"  # Alkene
        
    def condition_depth(self, d):
        """Check all reactions in the route for incompatible protecting group usage"""
        reactions = self.get_rxns(d)
        
        has_incompatible_strategy = False
        
        for rxn in reactions:
            # Check if this reaction involves benzyl ether protection
            if self.is_benzyl_ether_protection(rxn):
                # Check if the molecule contains incompatible functional groups
                if self.has_incompatible_groups(rxn):
                    has_incompatible_strategy = True
                    break
        
        # Condition is met if we want to allow incompatible use and it's present,
        # or if we want to penalize it and it's absent
        condition_met = has_incompatible_strategy == self.allow_incompatible
        
        return condition_met, len(reactions)
    
    def is_benzyl_ether_protection(self, rxn):
        """Detect if reaction involves benzyl ether protection of tertiary alcohol"""
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in rxn_parts[0].split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in rxn_parts[1].split(".")]
            
            if not all(reactants) or not all(products):
                return False
            
            # Look for formation of benzyl ether (benzyl group appears in products but not reactants)
            reactant_has_benzyl_ether = any(mol.HasSubstructMatch(Chem.MolFromSmarts(self.benzyl_ether_pattern)) for mol in reactants)
            product_has_benzyl_ether = any(mol.HasSubstructMatch(Chem.MolFromSmarts(self.benzyl_ether_pattern)) for mol in products)
            
            # Protection reaction: benzyl ether appears in product but not in reactant
            return not reactant_has_benzyl_ether and product_has_benzyl_ether
            
        except:
            return False
    
    def has_incompatible_groups(self, rxn):
        """Check if molecule contains terminal alkynes or alkenes"""
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            # Check both reactants and products for incompatible groups
            all_mols = []
            for smi in rxn_parts[0].split(".") + rxn_parts[1].split("."):
                mol = Chem.MolFromSmiles(smi.strip())
                if mol:
                    all_mols.append(mol)
            
            terminal_alkyne_smarts = Chem.MolFromSmarts(self.terminal_alkyne_pattern)
            alkene_smarts = Chem.MolFromSmarts(self.alkene_pattern)
            
            for mol in all_mols:
                if mol.HasSubstructMatch(terminal_alkyne_smarts) or mol.HasSubstructMatch(alkene_smarts):
                    return True
                    
            return False
            
        except:
            return False
    
    def route_scoring(self, x):
        """Convert condition result to score (0-10 scale)"""
        if x < 0:
            return 0  # Condition never met
        else:
            return 10 * (1 - x)  # Earlier detection is worse (higher penalty)
