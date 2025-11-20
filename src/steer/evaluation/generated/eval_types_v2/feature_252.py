"""Generated evaluation code for: Early trichloroacetimidate installation before organometallic chemistry"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class TrichloroacetimidateStrategy(BaseScoring):
    """
    Evaluates if trichloroacetimidate protecting group is installed early 
    and before organometallic chemistry occurs in the route.
    """
    
    def __init__(self, config: Dict):
        self.protecting_group_smarts = config["parameters"]["protecting_group_smarts"]
        self.installation_step = config["parameters"]["installation_step"]
        self.total_steps = config["parameters"]["total_steps"]
        self.incompatible_reactions = config["parameters"]["incompatible_reaction_types"]
        self.pg_pattern = Chem.MolFromSmarts(self.protecting_group_smarts)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Installation never happens
        
        # Check if installation happens before the target step
        target_fraction = self.installation_step / self.total_steps
        
        if x <= target_fraction:
            # Early installation is good, score based on how early
            return 10 * (1 - x / target_fraction)
        else:
            # Late installation is penalized
            return max(0, 5 * (1 - (x - target_fraction) / (1 - target_fraction)))
    
    def hit_condition(self, d):
        """Check if this reaction installs the trichloroacetimidate group"""
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            rxn_parts = rxn_smiles.split(">>")
            
            if len(rxn_parts) != 2:
                return False
                
            reactants = rxn_parts[0]
            products = rxn_parts[1].split(".")
            
            # Check if any reactant contains the protecting group
            reactant_has_pg = False
            for r_smi in reactants.split("."):
                r_mol = Chem.MolFromSmiles(r_smi)
                if r_mol and r_mol.HasSubstructMatch(self.pg_pattern):
                    reactant_has_pg = True
                    break
            
            # Check if product contains the protecting group
            product_has_pg = False
            for p_smi in products:
                p_mol = Chem.MolFromSmiles(p_smi)
                if p_mol and p_mol.HasSubstructMatch(self.pg_pattern):
                    product_has_pg = True
                    break
            
            # Installation occurs if reactants don't have PG but product does
            if not reactant_has_pg and product_has_pg:
                # Additional check: ensure no incompatible reactions occur later
                return self._check_no_later_incompatible_reactions(d)
                
            return False
            
        except Exception:
            return False
    
    def _check_no_later_incompatible_reactions(self, installation_node):
        """Check that no organometallic reactions occur after PG installation"""
        try:
            # Get the route from installation point to end
            current = installation_node
            while hasattr(current, 'children') and current.children:
                for child in current.children:
                    if self._is_organometallic_reaction(child):
                        return False
                    current = child
            return True
        except Exception:
            return True  # If we can't check, assume it's okay
    
    def _is_organometallic_reaction(self, node):
        """Detect if a reaction involves organometallic chemistry"""
        try:
            metadata = node.get("metadata", {})
            
            # Check policy name for organometallic indicators
            policy_name = metadata.get("policy_name", "").lower()
            if "organometallic" in policy_name or "grignard" in policy_name or "lithium" in policy_name:
                return True
                
            # Check reaction SMILES for organometallic patterns
            rxn_smiles = metadata.get("mapped_reaction_smiles", "")
            organometallic_patterns = [
                "[Mg]",  # Grignard reagents
                "[Li]",  # Organolithium
                "[Zn]",  # Organozinc
                "[Cu]",  # Organocopper
                "[Pd]",  # Palladium catalysis
                "[Ni]"   # Nickel catalysis
            ]
            
            for pattern in organometallic_patterns:
                if pattern in rxn_smiles:
                    return True
                    
            return False
            
        except Exception:
            return False
