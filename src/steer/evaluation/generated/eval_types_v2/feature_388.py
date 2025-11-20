"""Generated evaluation code for: Ester functional group interconversion sequence"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EsterInterconversionSequence(MultiRxnCondBase):
    """
    Evaluates synthesis routes for the presence of consecutive ester functional group 
    interconversion reactions (saponification followed by esterification).
    
    Checks for:
    1. Saponification: ester -> carboxylic acid (ester bond breaking)
    2. Esterification: carboxylic acid -> ester (ester bond formation)
    3. Consecutive occurrence of these reactions in the specified order
    """
    
    def __init__(self, config):
        self.reaction_sequence = config["reaction_sequence"]
        self.consecutive = config.get("consecutive", True)
        self.ester_pattern = Chem.MolFromSmarts("[#6](=[#8])-[#8]-[#6]")  # C(=O)-O-C
        self.carboxylic_acid_pattern = Chem.MolFromSmarts("[#6](=[#8])-[#8]")  # C(=O)-O
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        if len(reactions) < 2:
            return False, len(reactions)
        
        # Find saponification and esterification reactions
        saponification_indices = []
        esterification_indices = []
        
        for i, rxn in enumerate(reactions):
            if self.detect_saponification(rxn):
                saponification_indices.append(i)
            elif self.detect_esterification(rxn):
                esterification_indices.append(i)
        
        # Check if we have both reaction types
        if not saponification_indices or not esterification_indices:
            return False, len(reactions)
        
        # Check for consecutive sequence if required
        if self.consecutive:
            sequence_found = self.check_consecutive_sequence(
                saponification_indices, esterification_indices
            )
        else:
            sequence_found = True  # Just need both present
        
        return sequence_found, len(reactions)
    
    def detect_saponification(self, rxn) -> bool:
        """Detect ester -> carboxylic acid conversion (ester bond breaking)"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
        
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        try:
            # Parse reactants and products
            reactant_mols = [Chem.MolFromSmiles(smi.strip()) for smi in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(smi.strip()) for smi in products.split(".")]
            
            if not all(reactant_mols) or not all(product_mols):
                return False
            
            # Check if reactants contain ester and products contain carboxylic acid
            has_ester_reactant = any(mol.HasSubstructMatch(self.ester_pattern) for mol in reactant_mols)
            has_acid_product = any(mol.HasSubstructMatch(self.carboxylic_acid_pattern) for mol in product_mols)
            
            # Additional check: should have fewer ester bonds in products than reactants
            reactant_ester_count = sum(len(mol.GetSubstructMatches(self.ester_pattern)) for mol in reactant_mols)
            product_ester_count = sum(len(mol.GetSubstructMatches(self.ester_pattern)) for mol in product_mols)
            
            return has_ester_reactant and has_acid_product and (product_ester_count < reactant_ester_count)
            
        except:
            return False
    
    def detect_esterification(self, rxn) -> bool:
        """Detect carboxylic acid -> ester conversion (ester bond formation)"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
        
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        try:
            # Parse reactants and products
            reactant_mols = [Chem.MolFromSmiles(smi.strip()) for smi in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(smi.strip()) for smi in products.split(".")]
            
            if not all(reactant_mols) or not all(product_mols):
                return False
            
            # Check if reactants contain carboxylic acid and products contain ester
            has_acid_reactant = any(mol.HasSubstructMatch(self.carboxylic_acid_pattern) for mol in reactant_mols)
            has_ester_product = any(mol.HasSubstructMatch(self.ester_pattern) for mol in product_mols)
            
            # Additional check: should have more ester bonds in products than reactants
            reactant_ester_count = sum(len(mol.GetSubstructMatches(self.ester_pattern)) for mol in reactant_mols)
            product_ester_count = sum(len(mol.GetSubstructMatches(self.ester_pattern)) for mol in product_mols)
            
            return has_acid_reactant and has_ester_product and (product_ester_count > reactant_ester_count)
            
        except:
            return False
    
    def check_consecutive_sequence(self, saponification_indices, esterification_indices) -> bool:
        """Check if saponification is followed by esterification in consecutive steps"""
        for sap_idx in saponification_indices:
            for est_idx in esterification_indices:
                if est_idx == sap_idx + 1:  # Esterification immediately follows saponification
                    return True
        return False
