"""Generated evaluation code for: Intramolecular piperidine ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class IntramolecularPiperidineFormation(BaseScoring):
    """
    Evaluates synthesis routes based on intramolecular piperidine ring formation.
    Checks if a piperidine ring (6-membered ring with nitrogen) is formed via 
    intramolecular cyclization rather than using pre-formed piperidine fragments.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "bool")
        self.target_depth = config.get("target_depth", {}).get("value", 0)
        self.piperidine_smarts = "C1CCCCN1"  # 6-membered ring with nitrogen
        
    def route_scoring(self, x) -> float:
        if self.condition_type == "bool":
            if x < 0:
                return 0  # Condition not met
            else:
                return 1  # Condition met
        else:
            if x < 0:
                return 0
            return max(0, 1 - abs(x - self.target_depth))
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction represents intramolecular piperidine formation
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
                
            reactants_smiles, product_smiles = rxn_smiles.split(">>")
            
            # Parse molecules
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".") if r.strip()]
            
            if not product or not reactants:
                return False
            
            # Check if product contains piperidine ring
            piperidine_pattern = Chem.MolFromSmarts(self.piperidine_smarts)
            if not product.HasSubstructMatch(piperidine_pattern):
                return False
            
            # Check that no reactant already contains a piperidine ring (intramolecular formation)
            for reactant in reactants:
                if reactant and reactant.HasSubstructMatch(piperidine_pattern):
                    return False
            
            # Additional check: verify this is a cyclization (ring count increases)
            product_rings = len(Chem.GetSymmSSSR(product))
            total_reactant_rings = sum(len(Chem.GetSymmSSSR(r)) for r in reactants if r)
            
            # Ring formation should increase ring count
            if product_rings <= total_reactant_rings:
                return False
            
            # Check for intramolecular nature: single reactant with appropriate chain length
            if len(reactants) == 1:
                reactant = reactants[0]
                # Look for nitrogen and appropriate carbon chain that could cyclize
                nitrogen_pattern = Chem.MolFromSmarts("[N]")
                if reactant.HasSubstructMatch(nitrogen_pattern):
                    return True
            
            return False
            
        except Exception:
            return False
