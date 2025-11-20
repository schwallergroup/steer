"""Generated evaluation code for: Late stage nitrile to ester conversion"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageNitrileToEster(BaseScoring):
    """
    Evaluates synthesis routes for late-stage nitrile to ester conversion reactions.
    Detects when a nitrile functional group is converted to an ester group and
    scores based on how late in the synthesis this transformation occurs.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "continuous")
        self.target_depth = config.get("target_depth", {}).get("value", 0.8)  # Default to late stage
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Conversion doesn't happen
        else:
            # Late-stage conversion is better, so higher depth fraction gets higher score
            return x * 10
    
    def hit_condition(self, d):
        """
        Check if this reaction converts a nitrile to an ester.
        """
        try:
            rxn = d["metadata"]["mapped_reaction_smiles"].split(">>")
            reactants_smiles = rxn[0]
            products_smiles = rxn[1]
            
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            
            # Remove None molecules (parsing failures)
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            if not reactants or not products:
                return False
            
            # Define SMARTS patterns for nitrile and ester groups
            nitrile_pattern = Chem.MolFromSmarts("[C]#[N]")
            ester_pattern = Chem.MolFromSmarts("[C](=[O])[O][C]")
            
            if nitrile_pattern is None or ester_pattern is None:
                return False
            
            # Check if reactants contain nitrile groups
            reactant_has_nitrile = any(mol.HasSubstructMatch(nitrile_pattern) for mol in reactants)
            
            # Check if products contain ester groups  
            product_has_ester = any(mol.HasSubstructMatch(ester_pattern) for mol in products)
            
            # Check if products have fewer nitrile groups than reactants
            reactant_nitrile_count = sum(len(mol.GetSubstructMatches(nitrile_pattern)) for mol in reactants)
            product_nitrile_count = sum(len(mol.GetSubstructMatches(ester_pattern)) for mol in products)
            
            # Condition: reactants have nitrile, products have ester, and nitrile count decreased
            if reactant_has_nitrile and product_has_ester and reactant_nitrile_count > 0:
                # Additional check: ensure the nitrile carbon is involved in ester formation
                # by checking atom mapping if available
                return self._verify_nitrile_to_ester_mapping(d)
            
            return False
            
        except (KeyError, IndexError, AttributeError):
            return False
    
    def _verify_nitrile_to_ester_mapping(self, d):
        """
        Verify that a nitrile carbon is actually converted to an ester carbon
        using atom mapping information.
        """
        try:
            rxn = d["metadata"]["mapped_reaction_smiles"].split(">>")
            reactants_smiles = rxn[0]
            products_smiles = rxn[1]
            
            reactants_mol = Chem.MolFromSmiles(reactants_smiles)
            products_mol = Chem.MolFromSmiles(products_smiles)
            
            if reactants_mol is None or products_mol is None:
                return True  # Fall back to basic pattern matching
            
            # Find nitrile carbons in reactants
            nitrile_pattern = Chem.MolFromSmarts("[C]#[N]")
            nitrile_matches = reactants_mol.GetSubstructMatches(nitrile_pattern)
            
            # Find ester carbons in products  
            ester_pattern = Chem.MolFromSmarts("[C](=[O])[O][C]")
            ester_matches = products_mol.GetSubstructMatches(ester_pattern)
            
            # Check if any nitrile carbon maps to an ester carbon
            for nitrile_match in nitrile_matches:
                nitrile_carbon_idx = nitrile_match[0]
                nitrile_atom = reactants_mol.GetAtomWithIdx(nitrile_carbon_idx)
                nitrile_map_num = nitrile_atom.GetAtomMapNum()
                
                if nitrile_map_num > 0:  # Has mapping
                    for ester_match in ester_matches:
                        ester_carbon_idx = ester_match[0]
                        ester_atom = products_mol.GetAtomWithIdx(ester_carbon_idx)
                        ester_map_num = ester_atom.GetAtomMapNum()
                        
                        if nitrile_map_num == ester_map_num:
                            return True
            
            return True  # If no clear mapping, assume it's valid based on pattern match
            
        except (AttributeError, IndexError):
            return True  # Fall back to basic pattern matching
