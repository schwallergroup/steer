"""Generated evaluation code for: Late stage ester hydrolysis with base sensitive groups"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageEsterHydrolysisBaseSensitive(BaseScoring):
    """
    Evaluates routes for late-stage ester hydrolysis reactions performed in the presence 
    of base-sensitive groups like phthalimide, creating selectivity challenges.
    
    Rewards routes where ester hydrolysis occurs late in the synthesis while 
    base-sensitive groups are present, indicating challenging selectivity control.
    """
    
    def __init__(self, config):
        self.sensitive_groups = config.get("sensitive_groups", ["phthalimide"])
        self.phthalimide_pattern = "[#6]1[#6][#6]2[#6](=[#8])[#7]([#1,#6])[#6](=[#8])[#6]2[#6][#6]1"
        
    def route_scoring(self, x):
        if x < 0:
            return 0  # Condition not met
        else:
            # Higher score for later stage (closer to 1.0)
            # Scale to 0-10 range, favoring late-stage occurrence
            return x * 10
    
    def hit_condition(self, d):
        """Check if this reaction is an ester hydrolysis with base-sensitive groups present."""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn:
            return False
            
        # Split reaction SMILES
        rxn_parts = mapped_rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        product_smiles = rxn_parts[0]
        reactant_smiles = rxn_parts[1]
        
        try:
            product_mol = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r) for r in reactant_smiles.split(".")]
            
            if not product_mol or not all(reactants):
                return False
                
            # Check if this is an ester hydrolysis reaction
            if not self._is_ester_hydrolysis(product_mol, reactants):
                return False
                
            # Check if base-sensitive groups are present in the product
            return self._has_base_sensitive_groups(product_mol)
            
        except:
            return False
    
    def _is_ester_hydrolysis(self, product_mol, reactants):
        """Detect ester hydrolysis by looking for ester -> carboxylic acid + alcohol conversion."""
        # Ester pattern: R-COO-R'
        ester_pattern = Chem.MolFromSmarts("[#6][#6](=[#8])[#8][#6]")
        # Carboxylic acid pattern: R-COOH
        carb_acid_pattern = Chem.MolFromSmarts("[#6][#6](=[#8])[#8][#1]")
        
        # Check if reactants contain ester
        has_ester_reactant = any(mol.HasSubstructMatch(ester_pattern) for mol in reactants if mol)
        
        # Check if product contains carboxylic acid
        has_carb_acid_product = product_mol.HasSubstructMatch(carb_acid_pattern)
        
        return has_ester_reactant and has_carb_acid_product
    
    def _has_base_sensitive_groups(self, mol):
        """Check for presence of base-sensitive groups in the molecule."""
        phthalimide_smarts = Chem.MolFromSmarts(self.phthalimide_pattern)
        
        if "phthalimide" in self.sensitive_groups:
            if mol.HasSubstructMatch(phthalimide_smarts):
                return True
                
        # Can extend for other base-sensitive groups
        return False
