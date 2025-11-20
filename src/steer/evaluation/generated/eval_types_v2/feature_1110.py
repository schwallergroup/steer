"""Generated evaluation code for: Boc protecting group for enamine nitrogen"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BocEnamineProtection(BaseScoring):
    """
    Evaluates synthesis routes for the presence of Boc protection of enamine nitrogen.
    Checks if a Boc protecting group is added to an amine in an enamine context.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "fraction")
        self.target_depth = config.get("target_depth", {}).get("value", 0.5)

    def route_scoring(self, x) -> float:
        """Convert depth fraction to 0-10 score"""
        if x < 0:
            return 0  # Protection reaction doesn't happen
        else:
            # Earlier protection is generally better
            if self.condition_type == "bool":
                return 10 if x >= 0 else 0
            else:
                return max(0, 10 - abs(x - self.target_depth) * 10)

    def hit_condition(self, d) -> bool:
        """Check if this reaction involves Boc protection of enamine nitrogen"""
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            reactants, products = rxn_smiles.split(">>")
            reactant_mols = [Chem.MolFromSmiles(smi) for smi in reactants.split(".")]
            product_mol = Chem.MolFromSmiles(products)
            
            if not all(reactant_mols) or not product_mol:
                return False
            
            # Check for Boc reagent in reactants
            boc_pattern = Chem.MolFromSmarts("[C](=O)OC(C)(C)C")  # Boc anhydride pattern
            boc_chloride_pattern = Chem.MolFromSmarts("ClC(=O)OC(C)(C)C")  # Boc-Cl pattern
            
            has_boc_reagent = any(
                mol.HasSubstructMatch(boc_pattern) or mol.HasSubstructMatch(boc_chloride_pattern)
                for mol in reactant_mols if mol
            )
            
            if not has_boc_reagent:
                return False
            
            # Check for enamine pattern in reactants and Boc-protected amine in products
            enamine_pattern = Chem.MolFromSmarts("N=C-C")  # Basic enamine pattern
            boc_amine_pattern = Chem.MolFromSmarts("NC(=O)OC(C)(C)C")  # Boc-protected amine
            
            # Find reactant with enamine
            enamine_reactant = None
            for mol in reactant_mols:
                if mol and mol.HasSubstructMatch(enamine_pattern):
                    enamine_reactant = mol
                    break
            
            if not enamine_reactant:
                return False
            
            # Check if product has Boc-protected amine
            if not product_mol.HasSubstructMatch(boc_amine_pattern):
                return False
            
            # Additional check: ensure the nitrogen that was part of enamine is now Boc-protected
            # by comparing atom map numbers if available
            enamine_matches = enamine_reactant.GetSubstructMatches(enamine_pattern)
            boc_matches = product_mol.GetSubstructMatches(boc_amine_pattern)
            
            if enamine_matches and boc_matches:
                # Check if any enamine nitrogen corresponds to Boc-protected nitrogen via atom mapping
                for enamine_match in enamine_matches:
                    enamine_n_atom = enamine_reactant.GetAtomWithIdx(enamine_match[0])
                    if enamine_n_atom.GetAtomMapNum() > 0:
                        for boc_match in boc_matches:
                            boc_n_atom = product_mol.GetAtomWithIdx(boc_match[0])
                            if (boc_n_atom.GetAtomMapNum() > 0 and 
                                enamine_n_atom.GetAtomMapNum() == boc_n_atom.GetAtomMapNum()):
                                return True
            
            # If no atom mapping, return True if we found both patterns and Boc reagent
            return True
            
        except Exception:
            return False
