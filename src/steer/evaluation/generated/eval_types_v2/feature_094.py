"""Generated evaluation code for: Late stage cyclopropanation reactions"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageCyclopropanation(BaseScoring):
    """
    Evaluates whether cyclopropanation reactions occur in the late stages of synthesis.
    
    A cyclopropanation reaction is detected by the formation of a cyclopropane ring
    (3-membered carbon ring) that wasn't present in the reactants.
    """
    
    def __init__(self, config: Dict):
        self.stage_cutoff = config.get("stage_cutoff", 0.3)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # No cyclopropanation found
        
        # Late stage is better - reward reactions occurring after the cutoff
        if x >= self.stage_cutoff:
            return 10  # Perfect score for late-stage cyclopropanation
        else:
            # Penalize early-stage cyclopropanation
            return 10 * (x / self.stage_cutoff)
    
    def hit_condition(self, d) -> bool:
        """
        Detect cyclopropanation by checking if a cyclopropane ring is formed.
        """
        rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
        if not rxn_smiles or ">>" not in rxn_smiles:
            return False
        
        try:
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse reactants and products
            reactant_mols = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            product_mols = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            # Filter out None molecules
            reactant_mols = [mol for mol in reactant_mols if mol is not None]
            product_mols = [mol for mol in product_mols if mol is not None]
            
            if not reactant_mols or not product_mols:
                return False
            
            # Count cyclopropane rings in reactants and products
            cyclopropane_pattern = Chem.MolFromSmarts("[C;R1]1[C;R1][C;R1]1")
            
            reactant_cyclopropanes = sum(len(mol.GetSubstructMatches(cyclopropane_pattern)) 
                                       for mol in reactant_mols)
            product_cyclopropanes = sum(len(mol.GetSubstructMatches(cyclopropane_pattern)) 
                                      for mol in product_mols)
            
            # Cyclopropanation occurs if more cyclopropane rings in products than reactants
            return product_cyclopropanes > reactant_cyclopropanes
            
        except Exception:
            return False
