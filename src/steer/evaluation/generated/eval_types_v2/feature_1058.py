"""Generated evaluation code for: Evans auxiliary cleavage strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EvansAuxiliaryCleavage(BaseScoring):
    """
    Evaluates synthesis routes for Evans auxiliary cleavage strategy.
    
    This class checks if the route employs Evans chiral auxiliary methodology
    with standard cleavage (typically H2O2/LiOH) to reveal the carboxylic acid.
    Earlier cleavage in the route receives a higher score.
    """
    
    def __init__(self, config: Dict):
        self.auxiliary_type = config.get("auxiliary_type", "evans_oxazolidinone")
        self.target_depth = config.get("target_depth", {})
        self.condition_type = self.target_depth.get("type", "depth")
        self.target_depth_value = self.target_depth.get("value", 0.5)

    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Cleavage doesn't happen
        
        if self.condition_type == "bool":
            return 1  # Found the cleavage reaction
        else:
            # Earlier cleavage (lower depth) gets higher score
            # Scale to 0-10 range where early cleavage scores higher
            return max(0, 10 * (1 - x))

    def hit_condition(self, d):
        """
        Check if this reaction represents Evans auxiliary cleavage.
        
        Looks for:
        1. Evans oxazolidinone auxiliary in reactant
        2. Carboxylic acid product formation
        3. Typical cleavage conditions (H2O2, LiOH patterns)
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        rxn_parts = mapped_rxn.split(">>")
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        # Parse molecules
        try:
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".")]
            reactant_mols = [mol for mol in reactant_mols if mol is not None]
            product_mols = [mol for mol in product_mols if mol is not None]
        except:
            return False
            
        if not reactant_mols or not product_mols:
            return False
            
        # Check for Evans oxazolidinone auxiliary in reactants
        evans_pattern = Chem.MolFromSmarts("[#6]1[#6][#7][#6](=O)[#8]1")  # Basic oxazolidinone
        evans_substituted = Chem.MolFromSmarts("C1C[N]C(=O)O1")  # More specific pattern
        
        has_evans_reactant = False
        for mol in reactant_mols:
            if mol.HasSubstructMatch(evans_pattern) or mol.HasSubstructMatch(evans_substituted):
                has_evans_reactant = True
                break
                
        if not has_evans_reactant:
            return False
            
        # Check for carboxylic acid formation in products
        carboxylic_acid_pattern = Chem.MolFromSmarts("[CX3](=O)[OX2H1]")
        has_carboxylic_product = False
        
        for mol in product_mols:
            if mol.HasSubstructMatch(carboxylic_acid_pattern):
                has_carboxylic_product = True
                break
                
        if not has_carboxylic_product:
            return False
            
        # Check that the Evans auxiliary is cleaved (not present in products)
        has_evans_product = False
        for mol in product_mols:
            if mol.HasSubstructMatch(evans_pattern) or mol.HasSubstructMatch(evans_substituted):
                has_evans_product = True
                break
                
        # Additional check for typical cleavage reagents (H2O2, LiOH patterns)
        has_cleavage_reagent = False
        cleavage_patterns = [
            "[H][O][O][H]",  # H2O2
            "[Li+].[OH-]",   # LiOH
            "[OH-]",         # General hydroxide
        ]
        
        for pattern_smarts in cleavage_patterns:
            try:
                pattern = Chem.MolFromSmarts(pattern_smarts)
                if pattern:
                    for mol in reactant_mols:
                        if mol.HasSubstructMatch(pattern):
                            has_cleavage_reagent = True
                            break
            except:
                continue
                
        # Return True if Evans auxiliary is present in reactants, 
        # carboxylic acid formed in products, auxiliary cleaved,
        # and typical cleavage conditions detected
        return (has_evans_reactant and 
                has_carboxylic_product and 
                not has_evans_product and
                has_cleavage_reagent)
