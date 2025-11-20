"""Generated evaluation code for: Trifluoroacetamide protecting group cycle for chemoselectivity"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class TrifluoroacetamideProtection(BaseScoring):
    """
    Evaluates the use of trifluoroacetamide (TFA) protecting group strategy.
    Checks if TFA protection is applied to secondary amines to prevent 
    competing nucleophilic reactions, particularly in Williamson ether synthesis.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "bool")
        self.target_depth = config.get("target_depth", {}).get("value", 0)
        
    def route_scoring(self, x) -> float:
        if self.condition_type == "bool":
            return 1 if x >= 0 else 0  # Reward if protection strategy is used
        else:
            if x < 0:
                return 0  # Strategy not found
            return max(0, 1 - abs(x - self.target_depth))  # Prefer target depth
    
    def hit_condition(self, d):
        """Check if this reaction involves TFA protection of secondary amine"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            products = rxn_parts[0]
            reactants = rxn_parts[1]
            
            # Parse molecules
            prod_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".")]
            react_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            
            # Filter None molecules
            prod_mols = [m for m in prod_mols if m is not None]
            react_mols = [m for m in react_mols if m is not None]
            
            # Check for TFA protection pattern
            return self._is_tfa_protection_reaction(react_mols, prod_mols)
            
        except Exception:
            return False
    
    def _is_tfa_protection_reaction(self, reactants, products):
        """Check if reaction involves TFA protection of secondary amine"""
        
        # TFA anhydride or TFA-Cl pattern
        tfa_reagent_patterns = [
            "FC(F)(F)C(=O)OC(=O)C(F)(F)F",  # TFA anhydride
            "FC(F)(F)C(=O)Cl",               # TFA chloride
        ]
        
        # Secondary amine pattern (not in ring with N-H)
        sec_amine_pattern = "[NH1;!R]([CH])[CH]"  # Secondary amine not in ring
        piperidine_amine_pattern = "[NH1]1CCCCC1"  # Piperidine NH
        
        # TFA amide product pattern
        tfa_amide_pattern = "FC(F)(F)C(=O)N"
        
        # Check if we have TFA reagent in reactants
        has_tfa_reagent = False
        for reactant in reactants:
            for pattern in tfa_reagent_patterns:
                tfa_mol = Chem.MolFromSmarts(pattern)
                if tfa_mol and reactant.HasSubstructMatch(tfa_mol):
                    has_tfa_reagent = True
                    break
        
        if not has_tfa_reagent:
            return False
        
        # Check if we have secondary amine in reactants
        has_sec_amine = False
        for reactant in reactants:
            sec_amine_mol = Chem.MolFromSmarts(sec_amine_pattern)
            pip_amine_mol = Chem.MolFromSmarts(piperidine_amine_pattern)
            
            if ((sec_amine_mol and reactant.HasSubstructMatch(sec_amine_mol)) or 
                (pip_amine_mol and reactant.HasSubstructMatch(pip_amine_mol))):
                has_sec_amine = True
                break
        
        if not has_sec_amine:
            return False
        
        # Check if we have TFA amide in products
        has_tfa_amide = False
        for product in products:
            tfa_amide_mol = Chem.MolFromSmarts(tfa_amide_pattern)
            if tfa_amide_mol and product.HasSubstructMatch(tfa_amide_mol):
                has_tfa_amide = True
                break
        
        return has_tfa_reagent and has_sec_amine and has_tfa_amide
