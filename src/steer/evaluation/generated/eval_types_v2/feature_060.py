"""Generated evaluation code for: Evans auxiliary attachment and immediate removal"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EvansAuxiliaryWaste(MultiRxnCondBase):
    """
    Detects wasteful Evans auxiliary usage where the auxiliary is attached 
    and removed without performing any stereochemical transformations.
    Penalizes routes that use Evans auxiliary inefficiently.
    """
    
    def __init__(self, config):
        self.evans_smarts = config["parameters"]["smarts"]  # "N1C(=O)OC[C@@H]1Cc1ccccc1"
        self.evans_pattern = Chem.MolFromSmarts(self.evans_smarts)
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        """
        Check if Evans auxiliary is attached and removed without productive use.
        Returns True if wasteful pattern is detected.
        """
        reactions = self.get_rxns(d)
        
        # Track Evans auxiliary through the route
        has_attachment = False
        has_removal = False
        has_stereo_use = False
        
        for i, rxn in enumerate(reactions):
            attachment = self.detect_evans_attachment(rxn)
            removal = self.detect_evans_removal(rxn)
            stereo_use = self.detect_stereochemical_use(rxn)
            
            if attachment:
                has_attachment = True
            if removal:
                has_removal = True
            if stereo_use and self.has_evans_auxiliary_present(rxn):
                has_stereo_use = True
        
        # Wasteful if auxiliary is both attached and removed without stereochemical use
        wasteful_pattern = has_attachment and has_removal and not has_stereo_use
        
        return wasteful_pattern, len(reactions)
    
    def detect_evans_attachment(self, rxn):
        """Detect if Evans auxiliary is being attached in this reaction."""
        try:
            reactants_smiles, product_smiles = rxn.split(">>")
            
            # Check if product has Evans auxiliary but reactants don't
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            
            if not product:
                return False
                
            product_has_evans = product.HasSubstructMatch(self.evans_pattern)
            reactants_have_evans = any(r and r.HasSubstructMatch(self.evans_pattern) for r in reactants if r)
            
            return product_has_evans and not reactants_have_evans
            
        except:
            return False
    
    def detect_evans_removal(self, rxn):
        """Detect if Evans auxiliary is being removed in this reaction."""
        try:
            reactants_smiles, product_smiles = rxn.split(">>")
            
            # Check if reactants have Evans auxiliary but product doesn't
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            
            if not product:
                return False
                
            product_has_evans = product.HasSubstructMatch(self.evans_pattern)
            reactants_have_evans = any(r and r.HasSubstructMatch(self.evans_pattern) for r in reactants if r)
            
            return reactants_have_evans and not product_has_evans
            
        except:
            return False
    
    def has_evans_auxiliary_present(self, rxn):
        """Check if Evans auxiliary is present in the reaction."""
        try:
            reactants_smiles, product_smiles = rxn.split(">>")
            
            # Check both reactants and products
            all_molecules = []
            all_molecules.extend([Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")])
            all_molecules.append(Chem.MolFromSmiles(product_smiles))
            
            return any(mol and mol.HasSubstructMatch(self.evans_pattern) for mol in all_molecules if mol)
            
        except:
            return False
    
    def detect_stereochemical_use(self, rxn):
        """
        Detect if stereochemical transformation is occurring.
        This includes aldol reactions, alkylations, or other stereoselective processes.
        """
        try:
            reactants_smiles, product_smiles = rxn.split(">>")
            
            # Look for common stereochemical transformation patterns
            # Aldol reaction pattern - formation of new C-C bond adjacent to carbonyl
            aldol_pattern = Chem.MolFromSmarts("[C:1](=[O:2])[C:3][C:4]")
            
            # Alkylation pattern - new C-C bond formation at alpha position
            alkylation_pattern = Chem.MolFromSmarts("[C:1](=[O:2])[C:3]([C:4])")
            
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check for new stereocenters or typical Evans auxiliary reactions
            product_matches_aldol = product.HasSubstructMatch(aldol_pattern)
            product_matches_alkylation = product.HasSubstructMatch(alkylation_pattern)
            
            # Simple heuristic: if new bonds are formed near carbonyls, likely stereochemical
            return product_matches_aldol or product_matches_alkylation
            
        except:
            return False
    
    def route_scoring(self, x):
        """
        Score the route based on wasteful Evans auxiliary usage.
        Higher penalty (lower score) for wasteful usage.
        """
        if x < 0:
            return 10  # No wasteful usage detected
        else:
            return max(0, 10 - 8 * x)  # Heavy penalty for wasteful usage
