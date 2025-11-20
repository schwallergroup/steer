"""Generated evaluation code for: Late stage nitrile to acetylated amine conversion"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class NitrileToAcetylatedAmineConversion(BaseScoring):
    """
    Evaluates synthesis routes for late-stage conversion of nitrile to acetylated amine.
    Checks if a nitrile group (C≡N) is converted to an N-acetylated aminomethyl group
    in the final synthetic steps.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "bool")
        self.target_depth = config.get("target_depth", {}).get("value", 0.9)  # Late stage preferred
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Transformation doesn't happen
        else:
            # Late-stage transformation is better (higher depth fraction preferred)
            return x * 10  # Scale to 0-10, rewarding later transformations
            
    def hit_condition(self, d) -> bool:
        """Check if this reaction converts a nitrile to acetylated amine"""
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            rxn = rxn_smiles.split(">>")
            
            if len(rxn) != 2:
                return False
                
            reactants = rxn[0]
            products = rxn[1].split(".")
            
            # Parse reactant molecule
            reactant_mol = Chem.MolFromSmiles(reactants)
            if reactant_mol is None:
                return False
                
            # Parse product molecules
            product_mols = []
            for prod_smiles in products:
                prod_mol = Chem.MolFromSmiles(prod_smiles)
                if prod_mol is not None:
                    product_mols.append(prod_mol)
                    
            if not product_mols:
                return False
                
            # Check for nitrile in reactant
            nitrile_pattern = Chem.MolFromSmarts("[C]#[N]")
            if not reactant_mol.HasSubstructMatch(nitrile_pattern):
                return False
                
            # Check for acetylated amine in products
            # N-acetylated aminomethyl: -CH2-NH-CO-CH3
            acetylated_amine_pattern = Chem.MolFromSmarts("[CH2]-[NH]-[C](=[O])-[CH3]")
            
            for prod_mol in product_mols:
                if prod_mol.HasSubstructMatch(acetylated_amine_pattern):
                    # Verify the transformation by checking atom mapping
                    return self._verify_nitrile_to_acetylamine_mapping(reactant_mol, prod_mol, rxn_smiles)
                    
            return False
            
        except Exception:
            return False
            
    def _verify_nitrile_to_acetylamine_mapping(self, reactant_mol, product_mol, rxn_smiles) -> bool:
        """Verify the nitrile carbon becomes the aminomethyl carbon via atom mapping"""
        try:
            # Extract atom map numbers from nitrile carbon in reactant
            nitrile_pattern = Chem.MolFromSmarts("[C]#[N]")
            nitrile_matches = reactant_mol.GetSubstructMatches(nitrile_pattern)
            
            if not nitrile_matches:
                return False
                
            # Get atom map number of nitrile carbon
            nitrile_carbon_idx = nitrile_matches[0][0]  # First carbon in first match
            nitrile_carbon = reactant_mol.GetAtomWithIdx(nitrile_carbon_idx)
            nitrile_map_num = nitrile_carbon.GetAtomMapNum()
            
            if nitrile_map_num == 0:
                return False
                
            # Check if this mapped carbon is part of acetylated amine in product
            acetylated_amine_pattern = Chem.MolFromSmarts("[CH2]-[NH]-[C](=[O])-[CH3]")
            amine_matches = product_mol.GetSubstructMatches(acetylated_amine_pattern)
            
            for match in amine_matches:
                aminomethyl_carbon_idx = match[0]  # CH2 carbon
                aminomethyl_carbon = product_mol.GetAtomWithIdx(aminomethyl_carbon_idx)
                
                if aminomethyl_carbon.GetAtomMapNum() == nitrile_map_num:
                    return True
                    
            return False
            
        except Exception:
            return False
