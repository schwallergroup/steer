"""Generated evaluation code for: Late carbazole ring formation via nitrene cyclization"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class CarbazoleRingFormation(BaseScoring):
    """
    Evaluates whether carbazole ring formation via nitrene cyclization occurs late in the synthesis.
    Detects formation of carbazole core structure and rewards later-stage formation.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]  # carbazole pattern
        self.timing = config["parameters"]["timing"]  # "late"
        self.direction = config["parameters"]["direction"]  # "formation"
        self.carbazole_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        
        if self.timing == "late":
            # Reward later formation - higher depth fraction is better
            return 1 - x  # x is depth fraction, so 1-x rewards late formation
        elif self.timing == "early":
            return x  # Early formation preferred
        else:
            return 0.5  # Neutral if timing not specified
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves carbazole ring formation via nitrene cyclization.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
        
        rxn_parts = mapped_rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants_smiles = rxn_parts[0]
        products_smiles = rxn_parts[1]
        
        # Parse reactants and products
        try:
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            # Filter out None molecules
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            if not reactants or not products:
                return False
        except:
            return False
        
        # Check if carbazole is formed (present in products but not reactants)
        reactants_have_carbazole = any(mol.HasSubstructMatch(self.carbazole_pattern) for mol in reactants)
        products_have_carbazole = any(mol.HasSubstructMatch(self.carbazole_pattern) for mol in products)
        
        if self.direction == "formation":
            # Carbazole should be formed in this step
            carbazole_formed = products_have_carbazole and not reactants_have_carbazole
        elif self.direction == "breaking":
            # Carbazole should be broken in this step
            carbazole_formed = reactants_have_carbazole and not products_have_carbazole
        else:
            return False
        
        if not carbazole_formed:
            return False
        
        # Additional check for nitrene cyclization pattern
        # Look for nitrogen-containing reactant that could undergo cyclization
        nitrene_precursor_found = False
        for reactant in reactants:
            if reactant.GetNumAtoms() > 5:  # Reasonable size check
                # Look for azide or other nitrene precursor patterns
                azide_pattern = Chem.MolFromSmarts("[N-]=[N+]=[N-]")  # azide
                nitro_pattern = Chem.MolFromSmarts("[N+](=O)[O-]")    # nitro (potential nitrene source)
                amine_pattern = Chem.MolFromSmarts("N")               # general nitrogen
                
                if (reactant.HasSubstructMatch(azide_pattern) or 
                    reactant.HasSubstructMatch(nitro_pattern) or
                    reactant.HasSubstructMatch(amine_pattern)):
                    nitrene_precursor_found = True
                    break
        
        return carbazole_formed and nitrene_precursor_found
