"""Generated evaluation code for: Late stage pyridine ring formation via Skraup"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageSkraupPyridineFormation(BaseScoring):
    """
    Evaluates synthesis routes for late-stage pyridine ring formation via Skraup reaction.
    Checks if a pyridine ring is formed using Skraup cyclization conditions late in the synthesis.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]  # "c1ccncc1" for pyridine
        self.timing = config["parameters"]["timing"]  # "late"
        self.direction = config["parameters"]["direction"]  # "formation"
        self.reaction_type = config["parameters"]["reaction_type"]  # "Skraup"
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen via Skraup
        else:
            # For late-stage timing, higher depth fraction is better
            # Scale to 0-10 range where 1.0 depth gets score of 10
            return x * 10
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction node represents Skraup pyridine formation"""
        metadata = d.get("metadata", {})
        
        # Check if reaction involves glycerol or other Skraup indicators
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        if not mapped_rxn:
            return False
            
        rxn_parts = mapped_rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants_smiles = rxn_parts[0]
        products_smiles = rxn_parts[1]
        
        try:
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            if not all(reactants) or not all(products):
                return False
            
            # Check for pyridine ring formation
            pyridine_pattern = Chem.MolFromSmarts(self.ring_smarts)
            if not pyridine_pattern:
                return False
                
            # Count pyridine rings in reactants vs products
            reactant_pyridines = sum(mol.HasSubstructMatch(pyridine_pattern) for mol in reactants)
            product_pyridines = sum(mol.HasSubstructMatch(pyridine_pattern) for mol in products)
            
            # Check if pyridine ring is formed (more in products than reactants)
            if product_pyridines <= reactant_pyridines:
                return False
                
            # Check for Skraup reaction indicators
            # Look for glycerol (C(C(CO)O)O) or similar polyol patterns
            glycerol_pattern = Chem.MolFromSmarts("C(C(CO)O)O")
            polyol_pattern = Chem.MolFromSmarts("[CH2][CH]([OH])[CH2]")
            
            has_skraup_reagent = any(
                mol.HasSubstructMatch(glycerol_pattern) or 
                mol.HasSubstructMatch(polyol_pattern)
                for mol in reactants
            )
            
            # Also check for aniline derivatives (aromatic amine starting material)
            aniline_pattern = Chem.MolFromSmarts("c1ccccc1N")
            aminobenzene_pattern = Chem.MolFromSmarts("c1ccc(N)cc1")
            
            has_aniline = any(
                mol.HasSubstructMatch(aniline_pattern) or
                mol.HasSubstructMatch(aminobenzene_pattern)
                for mol in reactants
            )
            
            return has_skraup_reagent and has_aniline
            
        except Exception:
            return False
