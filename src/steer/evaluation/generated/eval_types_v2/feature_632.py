"""Generated evaluation code for: Convergent synthesis via two fragments"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentSynthesis(BaseScoring):
    """
    Evaluates convergent synthesis strategy by detecting when two fragments 
    are coupled via amide bond formation. Returns higher scores for earlier 
    convergent steps in the synthesis route.
    """
    
    def __init__(self, config: Dict):
        self.fragment_count = config.get("fragment_count", 2)
        self.coupling_reaction = config.get("coupling_reaction", "amide_coupling")
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # No convergent coupling found
        else:
            return 1 - x  # Earlier convergent coupling is better
    
    def hit_condition(self, d) -> bool:
        """
        Detects amide coupling between two fragments by checking:
        1. Exactly 2 reactants are present
        2. An amide bond is formed between reactants
        3. One reactant contains carboxyl/acid chloride, other contains amine
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            product_smiles = rxn_parts[0]
            reactant_smiles = rxn_parts[1]
            
            # Split reactants
            reactants = reactant_smiles.split(".")
            
            # Must have exactly 2 reactants for convergent coupling
            if len(reactants) != self.fragment_count:
                return False
                
            product_mol = Chem.MolFromSmiles(product_smiles)
            reactant_mols = [Chem.MolFromSmiles(r) for r in reactants]
            
            if not product_mol or not all(reactant_mols):
                return False
                
            # Check for amide bond formation
            if self.coupling_reaction == "amide_coupling":
                return self._detect_amide_coupling(product_mol, reactant_mols)
                
        except Exception:
            return False
            
        return False
    
    def _detect_amide_coupling(self, product_mol, reactant_mols) -> bool:
        """
        Detects amide coupling by checking:
        1. Product contains amide bond that wasn't in either reactant
        2. One reactant has carboxyl/acid chloride pattern
        3. Other reactant has amine pattern
        """
        # Amide bond pattern
        amide_pattern = Chem.MolFromSmarts("[C](=O)[NH]")
        if not amide_pattern:
            return False
            
        # Check if product has amide bond
        if not product_mol.HasSubstructMatch(amide_pattern):
            return False
            
        # Check that neither reactant already has this amide bond
        for reactant in reactant_mols:
            if reactant.HasSubstructMatch(amide_pattern):
                return False
                
        # Check for complementary functional groups
        carboxyl_patterns = [
            Chem.MolFromSmarts("[C](=O)[OH]"),  # carboxylic acid
            Chem.MolFromSmarts("[C](=O)[Cl]"),  # acid chloride
            Chem.MolFromSmarts("[C](=O)[O][C]") # ester (can couple under conditions)
        ]
        
        amine_pattern = Chem.MolFromSmarts("[NH2,NH1]") # primary or secondary amine
        
        if not amine_pattern:
            return False
            
        # Check if we have one electrophile and one nucleophile
        has_electrophile = False
        has_nucleophile = False
        
        for reactant in reactant_mols:
            # Check for electrophilic component
            for pattern in carboxyl_patterns:
                if pattern and reactant.HasSubstructMatch(pattern):
                    has_electrophile = True
                    break
                    
            # Check for nucleophilic component  
            if reactant.HasSubstructMatch(amine_pattern):
                has_nucleophile = True
                
        return has_electrophile and has_nucleophile
