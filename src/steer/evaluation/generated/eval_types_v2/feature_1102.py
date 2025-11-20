"""Generated evaluation code for: Early chiral center installation via sulfamidate alkylation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SulfamidateAlkylation(BaseScoring):
    """
    Evaluates synthesis routes for early chiral center installation via sulfamidate alkylation.
    
    This feature checks for the presence of cyclic sulfamidate ring-opening reactions
    that establish stereochemistry early in the synthesis. Higher scores are given
    when this stereochemical control occurs earlier in the route.
    """
    
    def __init__(self, config: Dict):
        self.timing_preference = config.get("timing", "early")
        self.require_stereocontrol = config.get("stereochemical_control", True)
        
        # SMARTS pattern for cyclic sulfamidate (5 or 6-membered ring)
        self.sulfamidate_5_pattern = "[C,N]1[C,N][C,N][S](=O)(=O)[N]1"
        self.sulfamidate_6_pattern = "[C,N]1[C,N][C,N][C,N][S](=O)(=O)[N]1"
    
    def route_scoring(self, x) -> float:
        """
        Convert depth fraction to score (0-10).
        For early timing preference, lower depth (earlier) gives higher score.
        """
        if x < 0:
            return 0  # Reaction not found
        
        if self.timing_preference == "early":
            # Early is better - score decreases with depth
            return max(0, 10 * (1 - x))
        else:
            # Later timing - score increases with depth
            return min(10, 10 * x)
    
    def hit_condition(self, d) -> bool:
        """
        Check if a reaction involves sulfamidate alkylation with ring opening.
        """
        try:
            metadata = d.get("metadata", {})
            mapped_rxn = metadata.get("mapped_reaction_smiles", "")
            
            if not mapped_rxn or ">>" not in mapped_rxn:
                return False
            
            reactants_smiles, products_smiles = mapped_rxn.split(">>")
            
            # Parse reactants and products
            reactants = []
            for r_smi in reactants_smiles.split("."):
                mol = Chem.MolFromSmiles(r_smi.strip())
                if mol:
                    reactants.append(mol)
            
            products = []
            for p_smi in products_smiles.split("."):
                mol = Chem.MolFromSmiles(p_smi.strip())
                if mol:
                    products.append(mol)
            
            if not reactants or not products:
                return False
            
            # Check for sulfamidate in reactants
            has_sulfamidate_reactant = self._has_sulfamidate(reactants)
            
            # Check that sulfamidate ring is opened in products
            has_ring_opening = self._check_ring_opening(reactants, products)
            
            # Check for alkylation (new C-C or C-N bond formation)
            has_alkylation = self._check_alkylation(reactants, products)
            
            # Check for stereochemical control if required
            stereo_control = True
            if self.require_stereocontrol:
                stereo_control = self._has_stereochemical_control(products)
            
            return has_sulfamidate_reactant and has_ring_opening and has_alkylation and stereo_control
            
        except Exception:
            return False
    
    def _has_sulfamidate(self, molecules) -> bool:
        """Check if any molecule contains a cyclic sulfamidate pattern."""
        pattern_5 = Chem.MolFromSmarts(self.sulfamidate_5_pattern)
        pattern_6 = Chem.MolFromSmarts(self.sulfamidate_6_pattern)
        
        if not pattern_5 or not pattern_6:
            return False
        
        for mol in molecules:
            if mol.HasSubstructMatch(pattern_5) or mol.HasSubstructMatch(pattern_6):
                return True
        return False
    
    def _check_ring_opening(self, reactants, products) -> bool:
        """
        Check if sulfamidate ring is opened by comparing ring counts
        and looking for opened sulfamidate fragments.
        """
        # Count sulfamidate rings in reactants vs products
        reactant_sulfamidate_rings = sum(self._count_sulfamidate_rings(mol) for mol in reactants)
        product_sulfamidate_rings = sum(self._count_sulfamidate_rings(mol) for mol in products)
        
        # Ring opening should reduce the count
        if reactant_sulfamidate_rings <= product_sulfamidate_rings:
            return False
        
        # Look for opened sulfamidate pattern (sulfonamide with alkyl chain)
        opened_pattern = Chem.MolFromSmarts("[C,N][S](=O)(=O)[N][C]")
        for mol in products:
            if opened_pattern and mol.HasSubstructMatch(opened_pattern):
                return True
        
        return False
    
    def _count_sulfamidate_rings(self, mol) -> int:
        """Count the number of sulfamidate rings in a molecule."""
        pattern_5 = Chem.MolFromSmarts(self.sulfamidate_5_pattern)
        pattern_6 = Chem.MolFromSmarts(self.sulfamidate_6_pattern)
        
        count = 0
        if pattern_5:
            count += len(mol.GetSubstructMatches(pattern_5))
        if pattern_6:
            count += len(mol.GetSubstructMatches(pattern_6))
        
        return count
    
    def _check_alkylation(self, reactants, products) -> bool:
        """
        Check for alkylation by looking for new C-C or C-N bonds
        and presence of alkylating agents.
        """
        # Look for common alkylating agent patterns in reactants
        alkylating_patterns = [
            "[C][Cl,Br,I]",  # Alkyl halides
            "[C][O][S](=O)(=O)[C]",  # Alkyl tosylates/mesylates  
            "[C][O][C](=O)[C]",  # Alkyl acetates/esters
        ]
        
        has_alkylating_agent = False
        for pattern_smarts in alkylating_patterns:
            pattern = Chem.MolFromSmarts(pattern_smarts)
            if pattern:
                for mol in reactants:
                    if mol.HasSubstructMatch(pattern):
                        has_alkylating_agent = True
                        break
            if has_alkylating_agent:
                break
        
        return has_alkylating_agent
    
    def _has_stereochemical_control(self, products) -> bool:
        """
        Check if products contain stereocenters, indicating stereochemical control.
        """
        for mol in products:
            # Count chiral centers
            chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
            if len(chiral_centers) > 0:
                return True
        return False
