"""Generated evaluation code for: Methoxy protecting group strategy for quinolinone"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class MethoxyQuinolinoneProtection(BaseScoring):
    """
    Evaluates methoxy protecting group strategy for quinolinone synthesis.
    Checks if methoxy installation occurs at specified step and removal at another step.
    """
    
    def __init__(self, config: Dict):
        self.protecting_group = config["parameters"]["protecting_group"]
        self.functional_group = config["parameters"]["functional_group"]
        self.installation_step = config["parameters"]["installation_step"]
        self.removal_step = config["parameters"]["removal_step"]
        
        # SMARTS patterns
        self.quinolinone_pattern = Chem.MolFromSmarts("c1ccc2c(c1)C(=O)Nc1ccccc12")
        self.methoxy_quinoline_pattern = Chem.MolFromSmarts("COc1ccc2c(c1)C(=O)Nc1ccccc12")
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Strategy not found
        return 10 - x * 2  # Earlier implementation is better, max score 10
    
    def hit_condition(self, d):
        """Check if this reaction involves methoxy protection/deprotection of quinolinone"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            product_mol = Chem.MolFromSmiles(rxn_parts[0])
            reactant_smiles = rxn_parts[1].split(".")
            reactant_mols = [Chem.MolFromSmiles(smi) for smi in reactant_smiles if smi]
            
            if not product_mol or not reactant_mols:
                return False
            
            # Check for methoxy installation (quinolinone -> methoxy-quinoline)
            product_has_methoxy_quinoline = product_mol.HasSubstructMatch(self.methoxy_quinoline_pattern)
            reactant_has_quinolinone = any(mol.HasSubstructMatch(self.quinolinone_pattern) for mol in reactant_mols)
            
            if product_has_methoxy_quinoline and reactant_has_quinolinone:
                return True
            
            # Check for methoxy removal (methoxy-quinoline -> quinolinone)
            product_has_quinolinone = product_mol.HasSubstructMatch(self.quinolinone_pattern)
            reactant_has_methoxy_quinoline = any(mol.HasSubstructMatch(self.methoxy_quinoline_pattern) for mol in reactant_mols)
            
            if product_has_quinolinone and reactant_has_methoxy_quinoline:
                return True
                
        except Exception:
            return False
            
        return False
