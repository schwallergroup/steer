"""Generated evaluation code for: Late stage Suzuki coupling for aryl installation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageSuzukiCoupling(BaseScoring):
    """
    Evaluates whether a Suzuki coupling reaction occurs late in the synthesis route.
    Detects Suzuki coupling by identifying aryl boronic acid patterns and C-C bond formation.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "fraction")
        self.target_depth = config.get("target_depth", {}).get("value", 0.8)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Suzuki coupling doesn't happen
        else:
            # Late-stage is better, so higher depth fraction gives higher score
            return min(10, x * 12.5)  # Scale so 0.8 depth = 10 points
    
    def hit_condition(self, d):
        """Check if this reaction is a Suzuki coupling with aryl boronic acid."""
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
                
            reactants, products = rxn_smiles.split(">>")
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            product_mol = Chem.MolFromSmiles(products.strip())
            
            if not all(reactant_mols) or not product_mol:
                return False
            
            # Check for boronic acid pattern in reactants
            boronic_acid_pattern = Chem.MolFromSmarts("[cH1,c]B(O)O")  # Aryl boronic acid
            boronate_ester_pattern = Chem.MolFromSmarts("[cH1,c]B1OC(C)(C)C(C)(C)O1")  # Pinacol boronate
            
            has_boron_reagent = False
            for mol in reactant_mols:
                if mol.HasSubstructMatch(boronic_acid_pattern) or mol.HasSubstructMatch(boronate_ester_pattern):
                    has_boron_reagent = True
                    break
            
            if not has_boron_reagent:
                return False
            
            # Check for aryl halide pattern
            aryl_halide_pattern = Chem.MolFromSmarts("[c][Cl,Br,I]")
            has_aryl_halide = False
            for mol in reactant_mols:
                if mol.HasSubstructMatch(aryl_halide_pattern):
                    has_aryl_halide = True
                    break
            
            if not has_aryl_halide:
                return False
            
            # Additional check: verify C-C bond formation between aromatic carbons
            # Count aromatic carbons in reactants vs products
            reactant_ar_carbons = sum(sum(1 for atom in mol.GetAtoms() 
                                        if atom.GetSymbol() == 'C' and atom.GetIsAromatic()) 
                                    for mol in reactant_mols)
            product_ar_carbons = sum(1 for atom in product_mol.GetAtoms() 
                                   if atom.GetSymbol() == 'C' and atom.GetIsAromatic())
            
            # In Suzuki coupling, we expect same number of aromatic carbons but new C-C bonds
            if reactant_ar_carbons != product_ar_carbons:
                return False
            
            # Check for biaryl pattern in product (indicating successful coupling)
            biaryl_pattern = Chem.MolFromSmarts("c-c")
            return product_mol.HasSubstructMatch(biaryl_pattern)
            
        except Exception:
            return False
