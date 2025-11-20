"""Generated evaluation code for: Sonogashira coupling for alkyne installation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SonogashiraCoupling(BaseScoring):
    """
    Evaluates routes based on the presence and timing of Sonogashira coupling reactions.
    Detects reactions that couple terminal alkynes with aryl/vinyl halides.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "bool")
        self.target_depth = config.get("target_depth", {}).get("value", -1)
        self.reagent_pattern = config["parameters"].get("reagent_pattern", "C#C")
        self.electrophile_pattern = config["parameters"].get("electrophile_pattern", "[Br,I]")

    def route_scoring(self, x) -> float:
        if self.condition_type == "bool":
            if self.target_depth == -1:  # Positive if condition not met
                return 1 if x < 0 else 0
        else:
            if x < 0:
                return 0
            return abs(x - self.target_depth)

    def hit_condition(self, d):
        """
        Check if a reaction node represents a Sonogashira coupling.
        Looks for alkyne + halide -> alkyne product pattern.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            product_smiles = rxn_parts[0]
            reactant_smiles = rxn_parts[1]
            
            product_mol = Chem.MolFromSmiles(product_smiles)
            if not product_mol:
                return False
                
            # Check if product contains alkyne
            alkyne_pattern = Chem.MolFromSmarts(self.reagent_pattern)
            if not product_mol.HasSubstructMatch(alkyne_pattern):
                return False
            
            # Parse reactants
            reactant_mols = []
            for r_smiles in reactant_smiles.split("."):
                mol = Chem.MolFromSmiles(r_smiles.strip())
                if mol:
                    reactant_mols.append(mol)
            
            if len(reactant_mols) < 2:
                return False
            
            # Check for alkyne and halide in reactants
            has_alkyne_reactant = False
            has_halide_reactant = False
            
            alkyne_reactant_pattern = Chem.MolFromSmarts(self.reagent_pattern)
            halide_pattern = Chem.MolFromSmarts(self.electrophile_pattern)
            
            for mol in reactant_mols:
                if mol.HasSubstructMatch(alkyne_reactant_pattern):
                    has_alkyne_reactant = True
                if mol.HasSubstructMatch(halide_pattern):
                    has_halide_reactant = True
            
            # Sonogashira: alkyne + halide -> coupled alkyne product
            return has_alkyne_reactant and has_halide_reactant
            
        except Exception:
            return False
