"""Generated evaluation code for: Trifluoroacetyl protecting group for piperidine amine"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class TrifluoroacetylPiperidineProtection(BaseScoring):
    """
    Evaluates synthesis routes for the use of trifluoroacetyl protecting group 
    on piperidine secondary amine. Checks at what depth the protection strategy 
    is employed in the synthetic route.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "bool")
        self.target_depth = config.get("target_depth", {}).get("value", -1)
        self.piperidine_pattern = "[#7]1[#6][#6][#6][#6][#6]1"
        self.trifluoroacetyl_pattern = "[#7]C(=O)C(F)(F)F"
        
    def route_scoring(self, x) -> float:
        if self.condition_type == "bool":
            if self.target_depth == -1:  # Positive if condition not met
                return 1 if x < 0 else 0
        else:
            if x < 0:
                return 0
            return max(0, 1 - abs(x - self.target_depth) / 10)
    
    def hit_condition(self, d) -> bool:
        """
        Check if a reaction involves trifluoroacetyl protection of piperidine amine
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles")
        
        if not mapped_rxn:
            return False
            
        rxn_parts = mapped_rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        products = rxn_parts[0]
        reactants = rxn_parts[1]
        
        try:
            # Parse molecules
            prod_mol = Chem.MolFromSmiles(products)
            react_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".") if r.strip()]
            
            if not prod_mol or not all(react_mols):
                return False
            
            # Check if product contains both piperidine and trifluoroacetyl patterns
            piperidine_smarts = Chem.MolFromSmarts(self.piperidine_pattern)
            trifluoroacetyl_smarts = Chem.MolFromSmarts(self.trifluoroacetyl_pattern)
            
            if not (piperidine_smarts and trifluoroacetyl_smarts):
                return False
                
            prod_has_piperidine = prod_mol.HasSubstructMatch(piperidine_smarts)
            prod_has_tfa = prod_mol.HasSubstructMatch(trifluoroacetyl_smarts)
            
            # Check if reactants have unprotected piperidine
            reactant_has_unprotected_pip = False
            reactant_has_tfa_reagent = False
            
            for react_mol in react_mols:
                has_pip = react_mol.HasSubstructMatch(piperidine_smarts)
                has_tfa = react_mol.HasSubstructMatch(trifluoroacetyl_smarts)
                
                if has_pip and not has_tfa:
                    reactant_has_unprotected_pip = True
                    
                # Check for trifluoroacetic anhydride or trifluoroacetyl chloride
                tfa_anhydride = Chem.MolFromSmarts("C(=O)(C(F)(F)F)OC(=O)C(F)(F)F")
                tfa_chloride = Chem.MolFromSmarts("C(=O)(C(F)(F)F)Cl")
                
                if (tfa_anhydride and react_mol.HasSubstructMatch(tfa_anhydride)) or \
                   (tfa_chloride and react_mol.HasSubstructMatch(tfa_chloride)):
                    reactant_has_tfa_reagent = True
            
            # Protection reaction: unprotected piperidine + TFA reagent -> protected product
            return (prod_has_piperidine and prod_has_tfa and 
                   reactant_has_unprotected_pip and reactant_has_tfa_reagent)
                   
        except Exception:
            return False
