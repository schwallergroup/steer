"""Generated evaluation code for: Acetate protecting group strategy for primary alcohol"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class AcetateProtectingGroupStrategy(BaseScoring):
    """
    Evaluates synthesis routes for acetate protecting group strategy on primary alcohols.
    Checks for installation of acetate protection at specified depth and later removal.
    """
    
    def __init__(self, config: Dict):
        self.installation_step = config["parameters"]["installation_step"]
        self.removal_step = config["parameters"]["removal_step"]
        self.primary_alcohol_pattern = Chem.MolFromSmarts("[CH2]-[OH]")
        self.acetate_pattern = Chem.MolFromSmarts("[CH2]-[O]-C(=O)-[CH3]")
        self.current_search_type = "installation"  # Track what we're looking for
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Strategy not found
        # Prefer execution close to target steps
        if self.current_search_type == "installation":
            return max(0, 10 - abs(x * 10 - self.installation_step))
        else:  # removal
            return max(0, 10 - abs(x * 10 - self.removal_step))
    
    def hit_condition(self, d):
        """Check if this reaction represents acetate protection installation or removal"""
        rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
        if not rxn_smiles or ">>" not in rxn_smiles:
            return False
            
        reactants_smiles, products_smiles = rxn_smiles.split(">>")
        reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
        products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
        
        # Filter out None molecules
        reactants = [mol for mol in reactants if mol is not None]
        products = [mol for mol in products if mol is not None]
        
        if self.current_search_type == "installation":
            return self._is_acetate_installation(reactants, products)
        else:
            return self._is_acetate_removal(reactants, products)
    
    def _is_acetate_installation(self, reactants, products):
        """Check if reaction converts primary alcohol to acetate ester"""
        # Look for primary alcohol in reactants
        has_primary_alcohol = any(
            mol.HasSubstructMatch(self.primary_alcohol_pattern) for mol in reactants
        )
        
        # Look for acetate ester in products
        has_acetate_ester = any(
            mol.HasSubstructMatch(self.acetate_pattern) for mol in products
        )
        
        # Check for acetylating agent (acetic anhydride or acetyl chloride)
        acetylating_agents = [
            Chem.MolFromSmarts("CC(=O)OC(=O)C"),  # acetic anhydride
            Chem.MolFromSmarts("CC(=O)Cl")        # acetyl chloride
        ]
        
        has_acetylating_agent = any(
            any(mol.HasSubstructMatch(agent) for mol in reactants)
            for agent in acetylating_agents if agent is not None
        )
        
        return has_primary_alcohol and has_acetate_ester and has_acetylating_agent
    
    def _is_acetate_removal(self, reactants, products):
        """Check if reaction converts acetate ester back to primary alcohol"""
        # Look for acetate ester in reactants
        has_acetate_ester = any(
            mol.HasSubstructMatch(self.acetate_pattern) for mol in reactants
        )
        
        # Look for primary alcohol in products
        has_primary_alcohol = any(
            mol.HasSubstructMatch(self.primary_alcohol_pattern) for mol in products
        )
        
        # Check for saponification conditions (base like NaOH, KOH)
        base_patterns = [
            Chem.MolFromSmarts("[Na+].[OH-]"),  # NaOH
            Chem.MolFromSmarts("[K+].[OH-]"),   # KOH
            Chem.MolFromSmarts("[OH-]")         # general hydroxide
        ]
        
        has_base = any(
            any(mol.HasSubstructMatch(base) for mol in reactants)
            for base in base_patterns if base is not None
        )
        
        return has_acetate_ester and has_primary_alcohol and has_base
    
    def condition_depth(self, d):
        """Override to handle two-step protection-deprotection strategy"""
        # First search for installation
        self.current_search_type = "installation"
        installation_found, installation_depth = super().condition_depth(d)
        
        if installation_found:
            # Then search for removal
            self.current_search_type = "removal"
            removal_found, removal_depth = super().condition_depth(d)
            
            if removal_found and removal_depth < installation_depth:
                # Both steps found in correct order (removal at lower depth = later in synthesis)
                return True, (installation_depth + removal_depth) / 2
        
        return False, -1
