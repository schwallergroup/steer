"""Generated evaluation code for: Late stage Weinreb ketone formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageWeinrebKetone(BaseScoring):
    """
    Evaluates synthesis routes for late-stage Weinreb ketone formation.
    
    Checks if a Weinreb amide to ketone conversion occurs within the specified
    depth threshold from the final product. Weinreb amides react with organometallic
    reagents to form ketones without over-addition issues.
    """
    
    def __init__(self, config: Dict):
        self.depth_threshold = config["parameters"].get("depth_threshold", 2)
        self.timing = config["parameters"].get("timing", "late")
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Weinreb ketone formation doesn't occur
        
        if self.timing == "late":
            # Reward earlier occurrence (smaller depth fraction)
            # Scale to 0-10 with preference for very late stage
            if x <= 0.2:  # Within first 20% of route depth
                return 10
            elif x <= 0.4:  # Within first 40%
                return 8
            elif x <= 0.6:
                return 5
            else:
                return 2
        else:
            # For non-late timing, just reward occurrence
            return 8 if x >= 0 else 0
    
    def hit_condition(self, d) -> bool:
        """
        Detects Weinreb ketone formation by identifying:
        1. Weinreb amide (N-methoxy-N-methylamide) in reactants
        2. Ketone formation in products
        3. Loss of the Weinreb amide functionality
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
        
        try:
            reactants_smiles, products_smiles = mapped_rxn.split(">>")
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            
            # Filter out None molecules
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            if not reactants or not products:
                return False
            
            # Weinreb amide pattern: N-methoxy-N-methylamide
            weinreb_pattern = Chem.MolFromSmarts("[C](=[O])[N]([CH3])[O][CH3]")
            ketone_pattern = Chem.MolFromSmarts("[C](=[O])[C,c]")
            
            if weinreb_pattern is None or ketone_pattern is None:
                return False
            
            # Check if reactants contain Weinreb amide
            has_weinreb_reactant = any(mol.HasSubstructMatch(weinreb_pattern) for mol in reactants)
            
            # Check if products contain ketone
            has_ketone_product = any(mol.HasSubstructMatch(ketone_pattern) for mol in products)
            
            # Check if Weinreb amide is consumed (not present in major products)
            weinreb_consumed = True
            for prod in products:
                # Skip small molecules that might be byproducts
                if prod.GetNumAtoms() > 5 and prod.HasSubstructMatch(weinreb_pattern):
                    weinreb_consumed = False
                    break
            
            return has_weinreb_reactant and has_ketone_product and weinreb_consumed
            
        except Exception:
            return False
