"""Generated evaluation code for: Phthalimide protecting group for primary amine"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class PhthalimideProtectingGroup(BaseScoring):
    """
    Evaluates if phthalimide protecting group strategy is used for primary amines.
    Checks for phthalimide formation (Gabriel synthesis) and deprotection reactions.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "bool")
        self.target_depth = config.get("target_depth", {}).get("value", -1)
        
        # SMARTS patterns for phthalimide structures
        self.phthalimide_pattern = "[#6]1:[#6]:[#6]:[#6]2:[#6](:[#6]:1)[#6](=[#8])[#7]([#6]):[#6]:2=[#8]"
        self.phthalimide_anion_pattern = "[#6]1:[#6]:[#6]:[#6]2:[#6](:[#6]:1)[#6](=[#8])[#7-]:[#6]:2=[#8]"
        self.primary_amine_pattern = "[#6][#7H2]"
        
    def route_scoring(self, x) -> float:
        if self.condition_type == "bool":
            if self.target_depth == -1:  # Positive if condition not met
                return 1 if x < 0 else 0
            else:
                return 1 if x >= 0 else 0
        else:
            if x < 0:
                return 0
            return max(0, 1 - abs(x - self.target_depth) / 10)
    
    def hit_condition(self, d) -> bool:
        """Check if reaction involves phthalimide protection or deprotection."""
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
                
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            reactants = [Chem.MolFromSmiles(smi) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi) for smi in products_smiles.split(".")]
            
            # Filter out None molecules
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            if not reactants or not products:
                return False
            
            # Check for phthalimide formation (Gabriel synthesis)
            if self._is_phthalimide_formation(reactants, products):
                return True
                
            # Check for phthalimide deprotection
            if self._is_phthalimide_deprotection(reactants, products):
                return True
                
            return False
            
        except Exception:
            return False
    
    def _is_phthalimide_formation(self, reactants, products) -> bool:
        """Check if reaction forms phthalimide from primary amine."""
        # Look for phthalimide pattern in products
        phthalimide_mol = Chem.MolFromSmarts(self.phthalimide_pattern)
        phthalimide_anion_mol = Chem.MolFromSmarts(self.phthalimide_anion_pattern)
        primary_amine_mol = Chem.MolFromSmarts(self.primary_amine_pattern)
        
        has_phthalimide_product = any(mol.HasSubstructMatch(phthalimide_mol) for mol in products)
        has_primary_amine_reactant = any(mol.HasSubstructMatch(primary_amine_mol) for mol in reactants)
        has_phthalimide_reactant = any(mol.HasSubstructMatch(phthalimide_anion_mol) or 
                                     mol.HasSubstructMatch(phthalimide_mol) for mol in reactants)
        
        return has_phthalimide_product and (has_primary_amine_reactant or has_phthalimide_reactant)
    
    def _is_phthalimide_deprotection(self, reactants, products) -> bool:
        """Check if reaction removes phthalimide to reveal primary amine."""
        phthalimide_mol = Chem.MolFromSmarts(self.phthalimide_pattern)
        primary_amine_mol = Chem.MolFromSmarts(self.primary_amine_pattern)
        
        has_phthalimide_reactant = any(mol.HasSubstructMatch(phthalimide_mol) for mol in reactants)
        has_primary_amine_product = any(mol.HasSubstructMatch(primary_amine_mol) for mol in products)
        
        # Check for hydrazine in reactants (common deprotection reagent)
        hydrazine_pattern = "[#7H2][#7H2]"
        hydrazine_mol = Chem.MolFromSmarts(hydrazine_pattern)
        has_hydrazine = any(mol.HasSubstructMatch(hydrazine_mol) for mol in reactants)
        
        return has_phthalimide_reactant and has_primary_amine_product and has_hydrazine
