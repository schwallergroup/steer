"""Generated evaluation code for: Late stage intramolecular cyclization to form morpholine"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateIntramolecularMorpholineFormation(BaseScoring):
    """
    Evaluates routes where morpholine ring formation occurs at late stages
    via intramolecular cyclization mechanism.
    """
    
    def __init__(self, config: Dict):
        self.morpholine_smarts = config["parameters"]["ring_smarts"]  # "C1COCCN1"
        self.timing = config["parameters"]["timing"]  # "late"
        self.mechanism = config["parameters"]["mechanism"]  # "intramolecular_alkylation"
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Morpholine formation doesn't occur
        else:
            # Late-stage cyclization is better, so invert the depth fraction
            return 1 - x
            
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction forms a morpholine ring via intramolecular cyclization
        """
        try:
            rxn = d["metadata"]["mapped_reaction_smiles"].split(">>")
            reactants_smiles = rxn[0]
            products_smiles = rxn[1]
            
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(smi) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi) for smi in products_smiles.split(".")]
            
            if not all(reactants) or not all(products):
                return False
                
            # Check if morpholine ring is formed (present in products but not reactants)
            morpholine_pattern = Chem.MolFromSmarts(self.morpholine_smarts)
            if not morpholine_pattern:
                return False
                
            # Check products contain morpholine
            product_has_morpholine = any(mol.HasSubstructMatch(morpholine_pattern) for mol in products)
            if not product_has_morpholine:
                return False
                
            # Check reactants don't contain morpholine (ring formation)
            reactant_has_morpholine = any(mol.HasSubstructMatch(morpholine_pattern) for mol in reactants)
            if reactant_has_morpholine:
                return False
                
            # Check for intramolecular mechanism - look for precursor with both N and O
            # that could cyclize (single reactant containing both heteroatoms)
            for reactant in reactants:
                if self._is_morpholine_precursor(reactant):
                    return True
                    
            return False
            
        except Exception:
            return False
            
    def _is_morpholine_precursor(self, mol) -> bool:
        """
        Check if molecule is a suitable precursor for intramolecular morpholine formation
        """
        if not mol:
            return False
            
        # Look for pattern that could cyclize to form morpholine
        # e.g., N-alkyl chain with terminal halogen and ether oxygen
        precursor_patterns = [
            # N with alkyl chain ending in leaving group, with ether oxygen
            "[N;H1,H2][CH2][CH2][O][CH2][CH2][Cl,Br,I]",
            "[N;H1,H2][CH2][CH2][O][CH2][CH2][CH2][Cl,Br,I]",
            # Alternative patterns for different chain lengths
            "[N][CH2][CH2][O][CH2][CH2][Cl,Br,I]",
            "[Cl,Br,I][CH2][CH2][O][CH2][CH2][N]"
        ]
        
        for pattern_smarts in precursor_patterns:
            try:
                pattern = Chem.MolFromSmarts(pattern_smarts)
                if pattern and mol.HasSubstructMatch(pattern):
                    return True
            except:
                continue
                
        return False
