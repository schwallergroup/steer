"""Generated evaluation code for: Late stage primary amide formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageAmideFormation(BaseScoring):
    """
    Evaluates whether primary amide formation occurs late in the synthesis route.
    Detects amidation reactions where carboxylic acid (C(=O)O) is converted to 
    primary amide (C(=O)N) and scores based on timing preference for late-stage installation.
    """
    
    def __init__(self, config: Dict):
        self.substrate_pattern = Chem.MolFromSmarts("C(=O)[OH]")
        self.product_pattern = Chem.MolFromSmarts("C(=O)[NH2]")
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Amidation doesn't happen
        else:
            return 1 - x  # Late-stage amidation is better (higher score for lower depth fraction)
    
    def hit_condition(self, d):
        """
        Check if this reaction node represents primary amide formation from carboxylic acid.
        """
        metadata = d.get("metadata", {})
        if "mapped_reaction_smiles" not in metadata:
            return False
            
        rxn_smiles = metadata["mapped_reaction_smiles"]
        rxn_parts = rxn_smiles.split(">>")
        
        if len(rxn_parts) != 2:
            return False
            
        # Parse product and reactants
        product_smiles = rxn_parts[0]
        reactants_smiles = rxn_parts[1]
        
        try:
            product_mol = Chem.MolFromSmiles(product_smiles)
            if product_mol is None:
                return False
                
            # Check if product contains primary amide
            if not product_mol.HasSubstructMatch(self.product_pattern):
                return False
            
            # Check reactants for carboxylic acid
            reactant_mols = []
            for reactant_smiles in reactants_smiles.split("."):
                mol = Chem.MolFromSmiles(reactant_smiles)
                if mol is not None:
                    reactant_mols.append(mol)
            
            # Look for carboxylic acid in reactants
            has_carboxylic_acid = any(mol.HasSubstructMatch(self.substrate_pattern) 
                                    for mol in reactant_mols)
            
            if not has_carboxylic_acid:
                return False
            
            # Verify transformation: carboxylic acid -> primary amide
            # Check that the amide carbon comes from the carboxylic acid carbon
            return self._verify_amidation_transformation(product_mol, reactant_mols)
            
        except Exception:
            return False
    
    def _verify_amidation_transformation(self, product_mol, reactant_mols):
        """
        Verify that the primary amide in product corresponds to carboxylic acid in reactants
        using atom mapping numbers.
        """
        # Get amide carbons in product with their map numbers
        amide_matches = product_mol.GetSubstructMatches(self.product_pattern)
        product_amide_carbons = set()
        
        for match in amide_matches:
            carbon_idx = match[0]  # First atom in pattern C(=O)[NH2] is carbon
            carbon_atom = product_mol.GetAtomWithIdx(carbon_idx)
            if carbon_atom.GetAtomMapNum() > 0:
                product_amide_carbons.add(carbon_atom.GetAtomMapNum())
        
        # Get carboxylic acid carbons in reactants with their map numbers
        reactant_acid_carbons = set()
        
        for mol in reactant_mols:
            acid_matches = mol.GetSubstructMatches(self.substrate_pattern)
            for match in acid_matches:
                carbon_idx = match[0]  # First atom in pattern C(=O)[OH] is carbon
                carbon_atom = mol.GetAtomWithIdx(carbon_idx)
                if carbon_atom.GetAtomMapNum() > 0:
                    reactant_acid_carbons.add(carbon_atom.GetAtomMapNum())
        
        # Check if any amide carbon in product maps to acid carbon in reactants
        return len(product_amide_carbons.intersection(reactant_acid_carbons)) > 0
