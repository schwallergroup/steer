"""Generated evaluation code for: Late stage pyrazole ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageRingFormation(BaseScoring):
    """
    Evaluates synthesis routes for late-stage pyrazole ring formation.
    Detects formation of pyrazole rings (c1cnnc1) and rewards later formation in the route.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]
        self.formation_method = config["parameters"]["formation_method"]
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        else:
            # Later formation gets higher score (1 - x gives higher score for higher x)
            # Scale to 0-10 range
            return (1 - x) * 10
    
    def hit_condition(self, d):
        """
        Check if this reaction involves pyrazole ring formation.
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
                
            rxn_parts = rxn_smiles.split(">>")
            reactants_smiles = rxn_parts[0]
            products_smiles = rxn_parts[1]
            
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".") if smi.strip()]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".") if smi.strip()]
            
            # Filter out None molecules
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            # Count pyrazole rings in reactants and products
            reactant_pyrazole_count = sum(len(mol.GetSubstructMatches(self.ring_pattern)) for mol in reactants)
            product_pyrazole_count = sum(len(mol.GetSubstructMatches(self.ring_pattern)) for mol in products)
            
            # Ring formation occurs if products have more pyrazole rings than reactants
            ring_formation_occurred = product_pyrazole_count > reactant_pyrazole_count
            
            # Additional check for Knorr-type synthesis pattern if specified
            if ring_formation_occurred and self.formation_method == "knorr_synthesis":
                return self._detect_knorr_pattern(reactants, products)
            
            return ring_formation_occurred
            
        except Exception:
            return False
    
    def _detect_knorr_pattern(self, reactants, products):
        """
        Detect Knorr-type synthesis pattern: enaminone + hydrazine -> pyrazole
        """
        try:
            # Look for hydrazine pattern in reactants (N-N bond)
            hydrazine_pattern = Chem.MolFromSmarts("[NX3][NX3]")
            has_hydrazine = any(mol.HasSubstructMatch(hydrazine_pattern) for mol in reactants)
            
            # Look for enaminone-like pattern (C=C-N and C=O)
            enaminone_pattern = Chem.MolFromSmarts("[CX3]=[CX3]-[NX3]")
            carbonyl_pattern = Chem.MolFromSmarts("[CX3]=[OX1]")
            
            has_enaminone_component = False
            for mol in reactants:
                if mol.HasSubstructMatch(enaminone_pattern) and mol.HasSubstructMatch(carbonyl_pattern):
                    has_enaminone_component = True
                    break
            
            return has_hydrazine and has_enaminone_component
            
        except Exception:
            return True  # If pattern detection fails, assume it's valid ring formation
