"""Generated evaluation code for: Protection deprotection cycling without intervening chemistry"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ProtectionDeprotectionCycling(MultiRxnCondBase):
    """
    Detects protection-deprotection cycling without intervening chemistry.
    Checks if a protecting group is added and then removed within a specified
    number of steps without any chemistry that would require the protection.
    """
    
    def __init__(self, config):
        self.protection_pairs = config.get("protection_deprotection_pairs", ["boc"])
        self.max_intervening_steps = config.get("intervening_steps", 0)
        self.same_functional_group = config.get("same_functional_group", True)
        
        # Define protection/deprotection patterns
        self.protection_patterns = {
            "boc": {
                "protection": "[NH2,NH1][C](=O)OC(C)(C)C",  # Boc-protected amine
                "deprotection": "[NH2,NH1]",  # Free amine after deprotection
                "protecting_group": "C(=O)OC(C)(C)C"  # Boc group itself
            }
        }
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        """Check if protection-deprotection cycling occurs in the route."""
        reactions = self.get_rxns(d)
        
        # Look for protection-deprotection cycles
        has_cycling = self.detect_protection_cycling(reactions)
        
        return has_cycling, len(reactions)
    
    def detect_protection_cycling(self, reactions) -> bool:
        """Detect if there's a protection followed by deprotection cycle."""
        for pair_name in self.protection_pairs:
            if pair_name not in self.protection_patterns:
                continue
                
            patterns = self.protection_patterns[pair_name]
            
            # Find all protection and deprotection reactions
            protection_indices = []
            deprotection_indices = []
            
            for i, rxn in enumerate(reactions):
                if self.is_protection_reaction(rxn, patterns):
                    protection_indices.append(i)
                elif self.is_deprotection_reaction(rxn, patterns):
                    deprotection_indices.append(i)
            
            # Check for cycling: protection followed by deprotection
            for prot_idx in protection_indices:
                for deprot_idx in deprotection_indices:
                    if deprot_idx > prot_idx:
                        steps_between = deprot_idx - prot_idx - 1
                        
                        if steps_between <= self.max_intervening_steps:
                            # Check if intervening reactions require protection
                            if not self.requires_protection(reactions[prot_idx+1:deprot_idx], patterns):
                                return True
        
        return False
    
    def is_protection_reaction(self, rxn, patterns) -> bool:
        """Check if reaction introduces a protecting group."""
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants = [Chem.MolFromSmiles(s.strip()) for s in rxn_parts[0].split(".") if s.strip()]
            products = [Chem.MolFromSmiles(s.strip()) for s in rxn_parts[1].split(".") if s.strip()]
            
            if not all(reactants + products):
                return False
            
            # Check if protection pattern appears in products but not reactants
            prot_pattern = Chem.MolFromSmarts(patterns["protection"])
            if not prot_pattern:
                return False
            
            has_protected_product = any(mol.HasSubstructMatch(prot_pattern) for mol in products)
            has_protected_reactant = any(mol.HasSubstructMatch(prot_pattern) for mol in reactants)
            
            return has_protected_product and not has_protected_reactant
            
        except:
            return False
    
    def is_deprotection_reaction(self, rxn, patterns) -> bool:
        """Check if reaction removes a protecting group."""
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants = [Chem.MolFromSmiles(s.strip()) for s in rxn_parts[0].split(".") if s.strip()]
            products = [Chem.MolFromSmiles(s.strip()) for s in rxn_parts[1].split(".") if s.strip()]
            
            if not all(reactants + products):
                return False
            
            # Check if protection pattern disappears from reactants to products
            prot_pattern = Chem.MolFromSmarts(patterns["protection"])
            if not prot_pattern:
                return False
            
            has_protected_reactant = any(mol.HasSubstructMatch(prot_pattern) for mol in reactants)
            has_protected_product = any(mol.HasSubstructMatch(prot_pattern) for mol in products)
            
            return has_protected_reactant and not has_protected_product
            
        except:
            return False
    
    def requires_protection(self, intervening_reactions, patterns) -> bool:
        """Check if intervening reactions would require the protecting group."""
        if not intervening_reactions:
            return False
        
        # Define reaction conditions that typically require protection
        harsh_conditions = [
            "[Li]",  # Organolithium reagents
            "[Mg]",  # Grignard reagents
            "C(=O)Cl",  # Acid chlorides
            "[OH-]",  # Strong base
            "N(C)(C)C(C)C"  # Strong amines
        ]
        
        for rxn in intervening_reactions:
            try:
                rxn_parts = rxn.split(">>")
                if len(rxn_parts) != 2:
                    continue
                    
                reactants = rxn_parts[0]
                
                # Check for harsh reaction conditions
                for condition in harsh_conditions:
                    pattern = Chem.MolFromSmarts(condition)
                    if pattern:
                        for reactant_smiles in reactants.split("."):
                            mol = Chem.MolFromSmiles(reactant_smiles.strip())
                            if mol and mol.HasSubstructMatch(pattern):
                                return True
                                
            except:
                continue
        
        return False
