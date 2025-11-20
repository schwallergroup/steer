"""Generated evaluation code for: Early Sandmeyer reaction for halogen installation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlySandmeyerReaction(BaseScoring):
    """
    Evaluates whether a Sandmeyer reaction occurs early in the synthesis route.
    Sandmeyer reactions involve conversion of diazonium salts to halides or other groups.
    """
    
    def __init__(self, config: Dict):
        self.depth_threshold = config["parameters"].get("depth_threshold", 4)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Sandmeyer reaction doesn't occur
        
        # Convert depth fraction to actual depth and check if early enough
        if x <= (self.depth_threshold / 10.0):  # Assuming max depth ~10 for normalization
            return 10  # Perfect score for early Sandmeyer
        else:
            # Penalize late Sandmeyer reactions
            return max(0, 10 - (x * 10 - self.depth_threshold) * 2)
    
    def hit_condition(self, d):
        """
        Detect Sandmeyer reaction by looking for:
        1. Diazonium salt conversion patterns
        2. Aniline to halide transformation
        3. Nitrogen-containing aromatic reactant converting to halogenated product
        """
        metadata = d.get("metadata", {})
        if "mapped_reaction_smiles" not in metadata:
            return False
            
        rxn_smiles = metadata["mapped_reaction_smiles"]
        reactants_smiles, products_smiles = rxn_smiles.split(">>")
        
        try:
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(smi) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi) for smi in products_smiles.split(".")]
            
            if not all(reactants) or not all(products):
                return False
            
            # Look for characteristic Sandmeyer patterns
            return self._detect_sandmeyer_pattern(reactants, products)
            
        except Exception:
            return False
    
    def _detect_sandmeyer_pattern(self, reactants, products):
        """
        Detect Sandmeyer reaction patterns:
        - Aromatic amine (aniline) to aromatic halide
        - Diazonium salt intermediate patterns
        """
        # Patterns for aromatic amines and halides
        aniline_pattern = Chem.MolFromSmarts("c[NH2,NH3+]")
        aryl_halide_pattern = Chem.MolFromSmarts("c[F,Cl,Br,I]")
        diazonium_pattern = Chem.MolFromSmarts("c[N+]#N")
        
        # Check if we have aniline in reactants
        has_aniline = any(mol.HasSubstructMatch(aniline_pattern) for mol in reactants)
        
        # Check if we have diazonium salt in reactants (direct Sandmeyer)
        has_diazonium = any(mol.HasSubstructMatch(diazonium_pattern) for mol in reactants)
        
        # Check if we have aryl halide in products
        has_aryl_halide = any(mol.HasSubstructMatch(aryl_halide_pattern) for mol in products)
        
        # Sandmeyer reaction: (aniline OR diazonium) -> aryl halide
        if (has_aniline or has_diazonium) and has_aryl_halide:
            # Additional check: ensure nitrogen is lost (aniline N -> halide)
            return self._verify_nitrogen_to_halide_conversion(reactants, products)
        
        return False
    
    def _verify_nitrogen_to_halide_conversion(self, reactants, products):
        """
        Verify that an aromatic carbon-nitrogen bond becomes carbon-halogen
        """
        # Count aromatic nitrogens in reactants vs products
        aromatic_n_reactants = 0
        aromatic_n_products = 0
        
        for mol in reactants:
            for atom in mol.GetAtoms():
                if atom.GetSymbol() == 'N' and atom.GetIsAromatic():
                    aromatic_n_reactants += 1
        
        for mol in products:
            for atom in mol.GetAtoms():
                if atom.GetSymbol() == 'N' and atom.GetIsAromatic():
                    aromatic_n_products += 1
        
        # In Sandmeyer, we typically lose aromatic nitrogen
        return aromatic_n_reactants > aromatic_n_products
