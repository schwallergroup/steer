"""Generated evaluation code for: Early ketone protection as acetal"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class KetoneAcetalProtection(BaseScoring):
    """
    Evaluates synthesis routes for early ketone protection as acetal.
    Checks if ketones are protected as acetals (dimethyl acetal formation)
    and rewards earlier protection in the synthesis route.
    """
    
    def __init__(self, config: Dict):
        self.timing_preference = config.get("timing", "early")
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Protection doesn't happen
        else:
            if self.timing_preference == "early":
                return 1 - x  # Earlier protection is better
            else:
                return x  # Later protection is better
    
    def hit_condition(self, d) -> bool:
        """
        Detects if a ketone protection as acetal occurs in this reaction.
        Looks for formation of dimethyl acetal from ketone.
        """
        metadata = d.get("metadata", {})
        if "mapped_reaction_smiles" not in metadata:
            return False
            
        rxn_smiles = metadata["mapped_reaction_smiles"]
        rxn_parts = rxn_smiles.split(">>")
        
        if len(rxn_parts) != 2:
            return False
            
        # Parse reactants and products
        reactants_smiles = rxn_parts[0]
        products_smiles = rxn_parts[1]
        
        try:
            # Get all reactant molecules
            reactant_mols = []
            for r_smi in reactants_smiles.split("."):
                mol = Chem.MolFromSmiles(r_smi)
                if mol is not None:
                    reactant_mols.append(mol)
            
            # Get all product molecules  
            product_mols = []
            for p_smi in products_smiles.split("."):
                mol = Chem.MolFromSmiles(p_smi)
                if mol is not None:
                    product_mols.append(mol)
            
            # Check if ketone protection is occurring
            return self._is_ketone_acetal_protection(reactant_mols, product_mols)
            
        except Exception:
            return False
    
    def _is_ketone_acetal_protection(self, reactants, products):
        """
        Check if reaction involves ketone -> acetal transformation
        """
        # Ketone pattern (C=O not in amide/ester/acid)
        ketone_pattern = Chem.MolFromSmarts("[CX3]=[OX1]")
        
        # Acetal patterns - dimethyl acetal specifically
        acetal_pattern = Chem.MolFromSmarts("[CH1](OC)(OC)")  # Dimethyl acetal carbon
        
        # Check if reactants contain ketone
        has_ketone_reactant = any(mol.HasSubstructMatch(ketone_pattern) for mol in reactants)
        
        # Check if products contain acetal
        has_acetal_product = any(mol.HasSubstructMatch(acetal_pattern) for mol in products)
        
        # Additional check: look for methanol as reactant (common acetal formation)
        methanol_pattern = Chem.MolFromSmarts("CO")
        has_methanol = any(mol.HasSubstructMatch(methanol_pattern) for mol in reactants)
        
        # Must have ketone in reactants, acetal in products, and preferably methanol
        if has_ketone_reactant and has_acetal_product:
            # Extra validation: check atom mapping to ensure same carbon
            return self._validate_protection_mapping(reactants, products, ketone_pattern, acetal_pattern)
        
        return False
    
    def _validate_protection_mapping(self, reactants, products, ketone_pattern, acetal_pattern):
        """
        Validate that the same carbon atom is involved in ketone->acetal transformation
        using atom mapping numbers.
        """
        try:
            # Find mapped ketone carbons in reactants
            ketone_carbons = set()
            for mol in reactants:
                matches = mol.GetSubstructMatches(ketone_pattern)
                for match in matches:
                    ketone_carbon_idx = match[0]  # First atom in pattern is the carbon
                    atom = mol.GetAtomWithIdx(ketone_carbon_idx)
                    if atom.GetAtomMapNum() > 0:
                        ketone_carbons.add(atom.GetAtomMapNum())
            
            # Find mapped acetal carbons in products
            acetal_carbons = set()
            for mol in products:
                matches = mol.GetSubstructMatches(acetal_pattern)
                for match in matches:
                    acetal_carbon_idx = match[0]  # First atom in pattern is the carbon
                    atom = mol.GetAtomWithIdx(acetal_carbon_idx)
                    if atom.GetAtomMapNum() > 0:
                        acetal_carbons.add(atom.GetAtomMapNum())
            
            # Check if any ketone carbon became an acetal carbon
            return len(ketone_carbons.intersection(acetal_carbons)) > 0
            
        except Exception:
            # If mapping validation fails, fall back to structure-only check
            return True
