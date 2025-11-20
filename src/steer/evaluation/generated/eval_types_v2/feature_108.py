"""Generated evaluation code for: N-carboethoxy protecting group on pyrazole"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class PyrazoleCarboethoxyProtection(BaseScoring):
    """
    Evaluates synthesis routes based on the use of N-carboethoxy protecting groups on pyrazole nitrogens.
    
    This class checks if pyrazole rings have carboethoxy (ethyl carbamate) protecting groups on nitrogen atoms,
    which is a common synthetic strategy to control regioselectivity and prevent unwanted reactions.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "bool")
        self.target_depth = config.get("target_depth", {}).get("value", -1)
    
    def route_scoring(self, x) -> float:
        """Convert depth fraction to scoring value (0-10 scale)"""
        if self.condition_type == "bool":
            if self.target_depth == -1:  # Positive if condition met
                return 1 if x >= 0 else 0
        else:
            if x < 0:
                return 0  # Protection strategy not found
            return max(0, 1 - abs(x - self.target_depth))  # Closer to target depth is better
    
    def hit_condition(self, d) -> bool:
        """Check if a reaction involves N-carboethoxy protected pyrazole"""
        try:
            # Get mapped reaction SMILES
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            # Split into products and reactants
            products, reactants = rxn_smiles.split(">>")
            
            # Check both products and reactants for protected pyrazole
            all_mols = []
            
            # Add product molecules
            for prod_smiles in products.split("."):
                if prod_smiles.strip():
                    mol = Chem.MolFromSmiles(prod_smiles.strip())
                    if mol:
                        all_mols.append(mol)
            
            # Add reactant molecules
            for react_smiles in reactants.split("."):
                if react_smiles.strip():
                    mol = Chem.MolFromSmiles(react_smiles.strip())
                    if mol:
                        all_mols.append(mol)
            
            # Check each molecule for N-carboethoxy protected pyrazole
            return any(self._has_protected_pyrazole(mol) for mol in all_mols)
            
        except Exception:
            return False
    
    def _has_protected_pyrazole(self, mol) -> bool:
        """Check if molecule contains pyrazole with N-carboethoxy protection"""
        if not mol:
            return False
        
        # SMARTS pattern for N-carboethoxy protected pyrazole
        # Pyrazole ring with carboethoxy group on nitrogen: n1cc[nH]c1 or n1c[nH]cc1 with COC(=O)N
        protected_pyrazole_patterns = [
            # N1-carboethoxy pyrazole (5-membered ring, N at position 1)
            "[n;R1]1[c][c][n;H1][c]1[C](=[O])[O][CH2][CH3]",
            "[n;R1]1[c][n;H1][c][c]1[C](=[O])[O][CH2][CH3]",
            # Alternative patterns with different connectivity
            "[CH3][CH2][O][C](=[O])[n]1[c][c][n;H1][c]1",
            "[CH3][CH2][O][C](=[O])[n]1[c][n;H1][c][c]1",
            # More general pattern for ethyl carbamate on pyrazole nitrogen
            "[CH3][CH2][O][C](=[O])[n]1[c,n][c,n][c,n][c,n]1"
        ]
        
        # Check each pattern
        for pattern in protected_pyrazole_patterns:
            try:
                patt_mol = Chem.MolFromSmarts(pattern)
                if patt_mol and mol.HasSubstructMatch(patt_mol):
                    return True
            except:
                continue
        
        # Alternative approach: look for pyrazole and carboethoxy separately but connected
        pyrazole_pattern = "[n]1[c][c][n][c]1"  # General pyrazole pattern
        carboethoxy_pattern = "[CH3][CH2][O][C](=[O])[N]"  # Ethyl carbamate pattern
        
        try:
            pyrazole_mol = Chem.MolFromSmarts(pyrazole_pattern)
            carboethoxy_mol = Chem.MolFromSmarts(carboethoxy_pattern)
            
            if (pyrazole_mol and carboethoxy_mol and 
                mol.HasSubstructMatch(pyrazole_mol) and 
                mol.HasSubstructMatch(carboethoxy_mol)):
                
                # Both patterns found, likely indicates protected pyrazole
                return True
                
        except:
            pass
        
        return False
