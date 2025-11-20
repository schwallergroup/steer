"""Generated evaluation code for: Late stage nucleophilic aromatic substitution"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageNucArSub(BaseScoring):
    """
    Evaluates whether nucleophilic aromatic substitution (SNAr) occurs at a late stage.
    Returns higher scores when SNAr reactions happen closer to the final step.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "continuous")
        self.target_depth = config.get("target_depth", {}).get("value", 0.1)  # Late stage default
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # SNAr reaction doesn't occur
        else:
            # Late stage is better - higher score for lower depth fraction
            return 1 - x
    
    def hit_condition(self, d):
        """
        Detect nucleophilic aromatic substitution by identifying:
        1. Aromatic ring with electron-withdrawing groups
        2. Nucleophile attacking aromatic carbon
        3. Leaving group departure from aromatic ring
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
                
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse molecules
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            if not all(reactants) or not all(products):
                return False
            
            # Look for aromatic substitution patterns
            return self._detect_snar_pattern(reactants, products)
            
        except Exception:
            return False
    
    def _detect_snar_pattern(self, reactants, products):
        """
        Detect SNAr by looking for:
        - Aromatic electrophile with leaving group (Cl, Br, F, NO2)
        - Nucleophile (containing N, O, S with lone pairs)
        - Product with nucleophile attached to aromatic ring
        """
        # Common SNAr electrophile patterns (aromatic ring with electron-withdrawing groups)
        electrophile_patterns = [
            "[cH0:1]1[c:2][c:3][c:4]([N+](=O)[O-])[c:5][c:6]1[Cl,Br,F:7]",  # nitro + halogen
            "[cH0:1]1[c:2][c:3][c:4]([C](=O)[O,N,C])[c:5][c:6]1[Cl,Br,F:7]",  # carbonyl + halogen
            "[cH0:1]1[c:2][c:3][c:4]([S](=O)(=O)[O,N,C])[c:5][c:6]1[Cl,Br,F:7]",  # sulfonyl + halogen
            "[c:1]1[c:2][c:3][c:4][c:5][c:6]1[Cl,Br,F:7]",  # simple aryl halide
        ]
        
        # Common nucleophile patterns
        nucleophile_patterns = [
            "[N:8][H,C]",  # amine
            "[O:8][H,C]",  # alcohol/alkoxide
            "[S:8][H,C]",  # thiol/thiolate
            "[N:8]=[C]",   # enamine/imine
        ]
        
        # Check if we have electrophile + nucleophile → substituted product
        for reactant in reactants:
            # Check for electrophile
            for elec_pattern in electrophile_patterns:
                elec_match = reactant.HasSubstructMatch(Chem.MolFromSmarts(elec_pattern))
                if elec_match:
                    # Look for nucleophile in other reactants
                    for other_reactant in reactants:
                        if other_reactant == reactant:
                            continue
                        for nuc_pattern in nucleophile_patterns:
                            nuc_match = other_reactant.HasSubstructMatch(Chem.MolFromSmarts(nuc_pattern))
                            if nuc_match:
                                # Check if product has nucleophile attached to aromatic ring
                                if self._check_substitution_product(products, elec_pattern, nuc_pattern):
                                    return True
        
        return False
    
    def _check_substitution_product(self, products, elec_pattern, nuc_pattern):
        """
        Check if products contain the expected SNAr substitution product
        """
        # Look for aromatic ring with nucleophile attached (leaving group departed)
        substitution_patterns = [
            "[c:1]1[c:2][c:3][c:4][c:5][c:6]1[N,O,S:8]",  # nucleophile attached to aromatic ring
        ]
        
        for product in products:
            for sub_pattern in substitution_patterns:
                if product.HasSubstructMatch(Chem.MolFromSmarts(sub_pattern)):
                    return True
        
        return False
