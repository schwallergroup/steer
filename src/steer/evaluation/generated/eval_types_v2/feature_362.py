"""Generated evaluation code for: Late stage Sonogashira coupling for alkyne introduction"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageSonogashira(BaseScoring):
    """
    Evaluates if Sonogashira coupling occurs at late stage (within first few steps).
    Sonogashira coupling forms C-C bonds between aryl/vinyl halides and terminal alkynes.
    """
    
    def __init__(self, config: Dict):
        self.target_step_position = config["parameters"].get("step_position", 2)
        
    def route_scoring(self, x) -> float:
        """
        Score based on how close the reaction occurs to the target step position.
        Late stage (early in synthesis) gets higher scores.
        """
        if x < 0:
            return 0  # Sonogashira coupling doesn't occur
        
        # Convert depth fraction to actual step number (approximate)
        # Early steps have lower depth fractions
        if x <= 0.3:  # Very early stage
            return 10
        elif x <= 0.5:  # Early to mid stage  
            return 8
        elif x <= 0.7:  # Mid stage
            return 5
        else:  # Late stage
            return 2
            
    def hit_condition(self, d) -> bool:
        """
        Detect Sonogashira coupling by checking for:
        1. Aryl/vinyl halide (I, Br, Cl) in reactants
        2. Terminal alkyne in reactants  
        3. Formation of new C-C bond between them
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
                
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse reactants
            reactant_mols = []
            for r_smiles in reactants_smiles.split("."):
                mol = Chem.MolFromSmiles(r_smiles)
                if mol:
                    reactant_mols.append(mol)
                    
            if len(reactant_mols) < 2:
                return False
                
            # Check for aryl/vinyl halide and terminal alkyne
            has_halide = False
            has_terminal_alkyne = False
            
            # SMARTS patterns
            aryl_iodide_pattern = Chem.MolFromSmarts("[cH0,cH1,cH2]I")  # Aryl iodide
            aryl_bromide_pattern = Chem.MolFromSmarts("[cH0,cH1,cH2]Br") # Aryl bromide
            vinyl_halide_pattern = Chem.MolFromSmarts("C=C[I,Br,Cl]")    # Vinyl halide
            terminal_alkyne_pattern = Chem.MolFromSmarts("C#C[H]")        # Terminal alkyne
            
            for mol in reactant_mols:
                # Check for halide
                if (mol.HasSubstructMatch(aryl_iodide_pattern) or 
                    mol.HasSubstructMatch(aryl_bromide_pattern) or
                    mol.HasSubstructMatch(vinyl_halide_pattern)):
                    has_halide = True
                    
                # Check for terminal alkyne
                if mol.HasSubstructMatch(terminal_alkyne_pattern):
                    has_terminal_alkyne = True
                    
            # Both components must be present
            if not (has_halide and has_terminal_alkyne):
                return False
                
            # Verify C-C bond formation by checking if product contains
            # the coupled alkyne structure (aryl/vinyl-alkyne)
            products_mol = Chem.MolFromSmiles(products_smiles)
            if not products_mol:
                return False
                
            # Pattern for aryl/vinyl-alkyne product
            coupled_alkyne_pattern = Chem.MolFromSmarts("[c,C]C#C")
            
            return products_mol.HasSubstructMatch(coupled_alkyne_pattern)
            
        except Exception:
            return False
