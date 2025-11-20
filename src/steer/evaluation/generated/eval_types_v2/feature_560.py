"""Generated evaluation code for: Late stage Suzuki coupling for biaryl formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageSuzuki(BaseScoring):
    """
    Evaluates synthesis routes for late-stage Suzuki coupling reactions.
    Scores routes higher when Suzuki coupling occurs near the end of the synthesis
    (after the stage_threshold fraction of the route is complete).
    """
    
    def __init__(self, config: Dict):
        self.stage_threshold = config["parameters"].get("stage_threshold", 0.8)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # No Suzuki coupling found
        
        # x is the depth fraction where Suzuki coupling occurs
        # We want late-stage (high x values) to score higher
        if x >= self.stage_threshold:
            return 10  # Perfect score for truly late-stage
        else:
            # Linear scaling: earlier reactions get lower scores
            return 10 * (x / self.stage_threshold)
    
    def hit_condition(self, d) -> bool:
        """
        Detects Suzuki coupling by looking for:
        1. Boronic acid/ester reactants
        2. Aryl halide reactants  
        3. Biaryl product formation
        """
        metadata = d.get("metadata", {})
        rxn_smiles = metadata.get("mapped_reaction_smiles", "")
        
        if not rxn_smiles or ">>" not in rxn_smiles:
            return False
            
        try:
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            reactant_mols = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            product_mols = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            if None in reactant_mols or None in product_mols:
                return False
            
            # Check for boronic acid/ester patterns
            boronic_acid_pattern = Chem.MolFromSmarts("[#6]-B(O)O")
            boronic_ester_pattern = Chem.MolFromSmarts("[#6]-B1OC(C)(C)CO1")  # Pinacol ester
            boronic_ester_pattern2 = Chem.MolFromSmarts("[#6]-B(OC)OC")  # Dimethyl ester
            
            # Check for aryl halide patterns
            aryl_halide_pattern = Chem.MolFromSmarts("c-[Cl,Br,I]")
            
            has_boronic_species = False
            has_aryl_halide = False
            
            for mol in reactant_mols:
                if (mol.HasSubstructMatch(boronic_acid_pattern) or 
                    mol.HasSubstructMatch(boronic_ester_pattern) or
                    mol.HasSubstructMatch(boronic_ester_pattern2)):
                    has_boronic_species = True
                    
                if mol.HasSubstructMatch(aryl_halide_pattern):
                    has_aryl_halide = True
            
            # Check for biaryl formation in products
            biaryl_pattern = Chem.MolFromSmarts("c-c")  # Aromatic C-C bond
            has_biaryl_product = any(mol.HasSubstructMatch(biaryl_pattern) for mol in product_mols)
            
            return has_boronic_species and has_aryl_halide and has_biaryl_product
            
        except Exception:
            return False
