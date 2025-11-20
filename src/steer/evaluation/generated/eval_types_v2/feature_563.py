"""Generated evaluation code for: Late stage Curtius rearrangement for amine protection"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageCarbamate(BaseScoring):
    """
    Evaluates synthesis routes for the presence of late-stage Curtius rearrangement
    to form Boc-protected amines from carboxylic acids.
    """
    
    def __init__(self, config: Dict):
        # Define SMARTS patterns for key components
        self.carboxylic_acid_pattern = Chem.MolFromSmarts("[CX3](=O)[OH]")
        self.boc_amine_pattern = Chem.MolFromSmarts("[NX3][C](=O)[O][C]([CH3])([CH3])[CH3]")
        self.dppa_pattern = Chem.MolFromSmarts("P(=O)([O][c]1[cH][cH][cH][cH][cH]1)([O][c]2[cH][cH][cH][cH][cH]2)[N]=[N+]=[N-]")
        self.tert_butanol_pattern = Chem.MolFromSmarts("[C]([CH3])([CH3])([CH3])[OH]")
        
        self.timing_preference = config.get("timing", "late")
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Curtius rearrangement doesn't happen
        else:
            # Late-stage (higher depth fraction) is better
            # Convert to 0-10 scale where late stage gets higher score
            return 10 * (1 - x)
    
    def hit_condition(self, d):
        """
        Check if this reaction represents a Curtius rearrangement with the specified reagents
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            reactants_smiles, products_smiles = mapped_rxn.split(">>")
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            # Filter out None molecules
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            if not reactants or not products:
                return False
            
            # Check for carboxylic acid to Boc-amine transformation
            has_carboxylic_acid_reactant = any(
                mol.HasSubstructMatch(self.carboxylic_acid_pattern) for mol in reactants
            )
            has_boc_amine_product = any(
                mol.HasSubstructMatch(self.boc_amine_pattern) for mol in products
            )
            
            # Check for required reagents (DPPA and tert-butanol)
            has_dppa = any(
                mol.HasSubstructMatch(self.dppa_pattern) for mol in reactants
            )
            has_tert_butanol = any(
                mol.HasSubstructMatch(self.tert_butanol_pattern) for mol in reactants
            )
            
            # Alternative check for tert-butanol as simple pattern
            if not has_tert_butanol:
                simple_tbutanol = Chem.MolFromSmarts("CC(C)(C)O")
                has_tert_butanol = any(
                    mol.HasSubstructMatch(simple_tbutanol) for mol in reactants
                )
            
            # Curtius rearrangement: carboxylic acid + DPPA + tert-butanol -> Boc-amine
            return (has_carboxylic_acid_reactant and 
                   has_boc_amine_product and 
                   has_dppa and 
                   has_tert_butanol)
                   
        except Exception as e:
            return False
