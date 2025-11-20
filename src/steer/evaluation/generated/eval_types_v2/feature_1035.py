"""Generated evaluation code for: Boc protecting group cycling strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BocProtectingGroupCycling(MultiRxnCondBase):
    """
    Evaluates routes for Boc protecting group cycling strategy.
    Checks if the route involves Boc deprotection followed by re-protection
    of the same nitrogen position.
    """
    
    def __init__(self, config):
        self.protecting_group = config.get("protecting_group", "Boc")
        self.pattern = config.get("pattern", "deprotection_reprotection")
        self.same_position = config.get("same_position", True)
        
        # Define Boc protecting group patterns
        self.boc_protected_n = Chem.MolFromSmarts("[N;!H0,!H1,!H2]C(=O)OC(C)(C)C")  # Boc-protected nitrogen
        self.free_amine = Chem.MolFromSmarts("[N;H1,H2]")  # Free amine
        
    def condition_depth(self, d):
        """
        Check if the route contains Boc deprotection followed by re-protection
        at the same nitrogen position.
        """
        reactions = self.get_rxns(d)
        
        # Track nitrogen atom map numbers through the route
        boc_deprotection_sites = set()
        boc_protection_sites = set()
        
        for rxn in reactions:
            deprotection_sites = self.detect_boc_deprotection(rxn)
            protection_sites = self.detect_boc_protection(rxn)
            
            boc_deprotection_sites.update(deprotection_sites)
            boc_protection_sites.update(protection_sites)
        
        # Check if there's overlap between deprotection and protection sites
        if self.same_position:
            cycling_sites = boc_deprotection_sites.intersection(boc_protection_sites)
            condition = len(cycling_sites) > 0
        else:
            condition = len(boc_deprotection_sites) > 0 and len(boc_protection_sites) > 0
        
        return condition, len(reactions)
    
    def detect_boc_deprotection(self, rxn):
        """
        Detect Boc deprotection reactions by finding Boc-protected nitrogens
        in reactants that become free amines in products.
        """
        deprotection_sites = set()
        
        try:
            rxn_parts = rxn.split(">>")
            reactants = [Chem.MolFromSmiles(r) for r in rxn_parts[0].split(".")]
            products = [Chem.MolFromSmiles(p) for p in rxn_parts[1].split(".")]
            
            # Find Boc-protected nitrogens in reactants
            reactant_boc_nitrogens = set()
            for mol in reactants:
                if mol is None:
                    continue
                matches = mol.GetSubstructMatches(self.boc_protected_n)
                for match in matches:
                    n_atom = mol.GetAtomWithIdx(match[0])
                    if n_atom.GetAtomMapNum() > 0:
                        reactant_boc_nitrogens.add(n_atom.GetAtomMapNum())
            
            # Find free amines in products
            product_free_amines = set()
            for mol in products:
                if mol is None:
                    continue
                matches = mol.GetSubstructMatches(self.free_amine)
                for match in matches:
                    n_atom = mol.GetAtomWithIdx(match[0])
                    if n_atom.GetAtomMapNum() > 0:
                        product_free_amines.add(n_atom.GetAtomMapNum())
            
            # Deprotection occurs when Boc-N becomes free amine
            deprotection_sites = reactant_boc_nitrogens.intersection(product_free_amines)
            
        except Exception:
            pass
        
        return deprotection_sites
    
    def detect_boc_protection(self, rxn):
        """
        Detect Boc protection reactions by finding free amines in reactants
        that become Boc-protected nitrogens in products.
        """
        protection_sites = set()
        
        try:
            rxn_parts = rxn.split(">>")
            reactants = [Chem.MolFromSmiles(r) for r in rxn_parts[0].split(".")]
            products = [Chem.MolFromSmiles(p) for p in rxn_parts[1].split(".")]
            
            # Find free amines in reactants
            reactant_free_amines = set()
            for mol in reactants:
                if mol is None:
                    continue
                matches = mol.GetSubstructMatches(self.free_amine)
                for match in matches:
                    n_atom = mol.GetAtomWithIdx(match[0])
                    if n_atom.GetAtomMapNum() > 0:
                        reactant_free_amines.add(n_atom.GetAtomMapNum())
            
            # Find Boc-protected nitrogens in products
            product_boc_nitrogens = set()
            for mol in products:
                if mol is None:
                    continue
                matches = mol.GetSubstructMatches(self.boc_protected_n)
                for match in matches:
                    n_atom = mol.GetAtomWithIdx(match[0])
                    if n_atom.GetAtomMapNum() > 0:
                        product_boc_nitrogens.add(n_atom.GetAtomMapNum())
            
            # Protection occurs when free amine becomes Boc-N
            protection_sites = reactant_free_amines.intersection(product_boc_nitrogens)
            
        except Exception:
            pass
        
        return protection_sites
