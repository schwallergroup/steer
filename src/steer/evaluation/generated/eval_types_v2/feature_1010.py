"""Generated evaluation code for: Orthogonal protecting group strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class OrthogonalProtectingGroupStrategy(MultiRxnCondBase):
    """
    Evaluates whether a synthesis route uses an orthogonal protecting group strategy.
    Checks for the presence of multiple protecting groups that can be removed under
    different conditions (e.g., TBDMS/fluoride, Boc/acid, PNB/reduction).
    """
    
    def __init__(self, config):
        self.protecting_groups = config.get("protecting_groups", ["TBDMS", "Boc", "PNB"])
        self.orthogonality = config.get("orthogonality", True)
        self.removal_conditions = config.get("removal_conditions", ["fluoride", "acid", "reduction"])
        
        # Define SMARTS patterns for protecting groups
        self.pg_patterns = {
            "TBDMS": "[Si](C)(C)(C(C)(C)C)",  # tert-butyldimethylsilyl
            "Boc": "C(=O)OC(C)(C)C",          # tert-butoxycarbonyl
            "PNB": "c1ccc(C[O])cc1[N+](=O)[O-]",  # para-nitrobenzyl
            "TBS": "[Si](C)(C)(C(C)(C)C)",     # alternative TBDMS pattern
            "Cbz": "C(=O)OCc1ccccc1",         # benzyloxycarbonyl
            "Fmoc": "C(=O)OCC1c2ccccc2-c2ccccc21"  # fluorenylmethoxycarbonyl
        }
        
        # Mapping of removal conditions to protecting groups
        self.condition_mapping = {
            "fluoride": ["TBDMS", "TBS"],
            "acid": ["Boc", "tBu"],
            "reduction": ["PNB", "Cbz"],
            "base": ["Fmoc"]
        }

    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Track protecting groups used and their removal conditions
        pg_used = set()
        removal_conditions_used = set()
        
        for rxn in reactions:
            # Check for protecting group installation
            pg_installed = self.detect_protection_reaction(rxn)
            if pg_installed:
                pg_used.add(pg_installed)
        
        for rxn in reactions:
            # Check for protecting group removal
            pg_removed, condition = self.detect_deprotection_reaction(rxn)
            if pg_removed and condition:
                removal_conditions_used.add(condition)
        
        # Check if we have orthogonal strategy
        required_pgs = set(self.protecting_groups)
        required_conditions = set(self.removal_conditions)
        
        if self.orthogonality:
            # Need multiple different protecting groups with different removal conditions
            has_multiple_pgs = len(pg_used.intersection(required_pgs)) >= 2
            has_orthogonal_conditions = len(removal_conditions_used.intersection(required_conditions)) >= 2
            condition_met = has_multiple_pgs and has_orthogonal_conditions
        else:
            # Just need the specified protecting groups to be present
            condition_met = required_pgs.issubset(pg_used)
        
        return condition_met, len(reactions)

    def detect_protection_reaction(self, rxn):
        """Detect if a reaction involves protecting group installation."""
        try:
            prod_smiles, react_smiles = rxn.split(">>")
            prod_mol = Chem.MolFromSmiles(prod_smiles)
            react_mols = [Chem.MolFromSmiles(r.strip()) for r in react_smiles.split(".")]
            
            if not prod_mol or not all(react_mols):
                return None
            
            # Check if product has protecting group that reactants don't have
            for pg_name, pattern in self.pg_patterns.items():
                pg_mol = Chem.MolFromSmarts(pattern)
                if not pg_mol:
                    continue
                    
                prod_has_pg = prod_mol.HasSubstructMatch(pg_mol)
                reactants_have_pg = any(mol.HasSubstructMatch(pg_mol) for mol in react_mols)
                
                # Protection: product has PG but reactants don't
                if prod_has_pg and not reactants_have_pg:
                    return pg_name
                    
            return None
        except:
            return None

    def detect_deprotection_reaction(self, rxn):
        """Detect if a reaction involves protecting group removal and determine conditions."""
        try:
            prod_smiles, react_smiles = rxn.split(">>")
            prod_mol = Chem.MolFromSmiles(prod_smiles)
            react_mols = [Chem.MolFromSmiles(r.strip()) for r in react_smiles.split(".")]
            
            if not prod_mol or not all(react_mols):
                return None, None
            
            # Check if reactant has protecting group that product doesn't have
            for pg_name, pattern in self.pg_patterns.items():
                pg_mol = Chem.MolFromSmarts(pattern)
                if not pg_mol:
                    continue
                    
                reactants_have_pg = any(mol.HasSubstructMatch(pg_mol) for mol in react_mols)
                prod_has_pg = prod_mol.HasSubstructMatch(pg_mol)
                
                # Deprotection: reactants have PG but product doesn't
                if reactants_have_pg and not prod_has_pg:
                    # Determine removal condition based on other reagents
                    condition = self.determine_removal_condition(react_smiles, pg_name)
                    return pg_name, condition
                    
            return None, None
        except:
            return None, None

    def determine_removal_condition(self, react_smiles, pg_name):
        """Determine removal condition based on reagents present."""
        react_smiles_lower = react_smiles.lower()
        
        # Look for characteristic reagents
        if any(reagent in react_smiles_lower for reagent in ["[f-]", "tbaf", "hf", "csf"]):
            return "fluoride"
        elif any(reagent in react_smiles_lower for reagent in ["tfa", "hcl", "[h+]", "acid"]):
            return "acid"  
        elif any(reagent in react_smiles_lower for reagent in ["[h]", "pd", "lialh4", "nabh4"]):
            return "reduction"
        elif any(reagent in react_smiles_lower for reagent in ["dbu", "piperidine", "[oh-]"]):
            return "base"
        
        # Fallback: map protecting group to typical removal condition
        for condition, pgs in self.condition_mapping.items():
            if pg_name in pgs:
                return condition
        
        return "unknown"
