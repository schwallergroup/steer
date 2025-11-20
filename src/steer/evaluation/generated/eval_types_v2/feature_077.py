"""Generated evaluation code for: SEM protecting group strategy for pyrazoles"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SEMProtectedPyrazoleStrategy(BaseScoring):
    """
    Evaluates synthesis routes for SEM protecting group strategy on pyrazole nitrogens.
    Checks if SEM groups are used to protect pyrazole nitrogens and counts occurrences.
    """
    
    def __init__(self, config: Dict):
        self.target_count = config["parameters"]["count"]
        self.sem_pattern = Chem.MolFromSmarts("[#6]-[#14](-[#6])(-[#6])-[#8]-[#6]-[#8]-[#6]")  # SEM group
        self.pyrazole_pattern = Chem.MolFromSmarts("c1n[nH]cc1")  # pyrazole core
        self.sem_pyrazole_pattern = Chem.MolFromSmarts("c1nn(-[CH2]-[#8]-[#14](-[#6])(-[#6])-[#6])cc1")  # SEM-protected pyrazole
        
    def route_scoring(self, x) -> float:
        """
        Score based on whether the target count of SEM-protected pyrazoles is found.
        x represents the count of SEM-protected pyrazole occurrences found.
        """
        if x < 0:
            return 0  # No SEM protection found
        
        # Score inversely proportional to deviation from target count
        deviation = abs(x - self.target_count)
        if deviation == 0:
            return 10  # Perfect match
        else:
            return max(0, 10 - deviation * 2)  # Penalize deviation
    
    def hit_condition(self, d) -> bool:
        """
        Check if a reaction involves SEM-protected pyrazole structures.
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles:
                return False
                
            parts = rxn_smiles.split(">>")
            if len(parts) != 2:
                return False
                
            products = parts[0]
            reactants = parts[1]
            
            # Check both products and reactants for SEM-protected pyrazoles
            all_molecules = []
            
            # Parse products
            for prod_smiles in products.split("."):
                mol = Chem.MolFromSmiles(prod_smiles)
                if mol:
                    all_molecules.append(mol)
                    
            # Parse reactants
            for react_smiles in reactants.split("."):
                mol = Chem.MolFromSmiles(react_smiles)
                if mol:
                    all_molecules.append(mol)
            
            # Count SEM-protected pyrazoles across all molecules
            sem_pyrazole_count = 0
            for mol in all_molecules:
                if mol.HasSubstructMatch(self.sem_pyrazole_pattern):
                    matches = mol.GetSubstructMatches(self.sem_pyrazole_pattern)
                    sem_pyrazole_count += len(matches)
                    
            return sem_pyrazole_count > 0
            
        except Exception:
            return False
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        """
        Override to count total SEM-protected pyrazoles in the reaction.
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles:
                return False, -1
                
            parts = rxn_smiles.split(">>")
            if len(parts) != 2:
                return False, -1
                
            products = parts[0]
            reactants = parts[1]
            
            # Count SEM-protected pyrazoles
            total_count = 0
            all_smiles = products.split(".") + reactants.split(".")
            
            for smiles in all_smiles:
                mol = Chem.MolFromSmiles(smiles)
                if mol and mol.HasSubstructMatch(self.sem_pyrazole_pattern):
                    matches = mol.GetSubstructMatches(self.sem_pyrazole_pattern)
                    total_count += len(matches)
            
            condition_met = total_count >= self.target_count
            return condition_met, total_count
            
        except Exception:
            return False, -1
