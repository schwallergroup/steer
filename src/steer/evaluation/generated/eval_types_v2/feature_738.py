"""Generated evaluation code for: Sequential amidoxime to oxadiazolone conversion"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SequentialAmidoximeOxadiazolone(MultiRxnCondBase):
    """
    Checks for sequential amidoxime to oxadiazolone conversion.
    Looks for a two-step sequence: nitrile -> amidoxime -> oxadiazolone.
    """
    
    def __init__(self, config):
        self.consecutive = config.get("consecutive", True)
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Find positions of each reaction type
        nitrile_to_amidoxime_positions = []
        amidoxime_to_oxadiazolone_positions = []
        
        for i, rxn in enumerate(reactions):
            if self.detect_nitrile_to_amidoxime(rxn):
                nitrile_to_amidoxime_positions.append(i)
            if self.detect_amidoxime_to_oxadiazolone(rxn):
                amidoxime_to_oxadiazolone_positions.append(i)
        
        # Check if both reaction types are present
        has_nitrile_to_amidoxime = len(nitrile_to_amidoxime_positions) > 0
        has_amidoxime_to_oxadiazolone = len(amidoxime_to_oxadiazolone_positions) > 0
        
        if not (has_nitrile_to_amidoxime and has_amidoxime_to_oxadiazolone):
            return False, len(reactions)
        
        # If consecutive required, check if reactions are sequential
        if self.consecutive:
            for nitrile_pos in nitrile_to_amidoxime_positions:
                for oxadiazolone_pos in amidoxime_to_oxadiazolone_positions:
                    if oxadiazolone_pos == nitrile_pos + 1:
                        return True, len(reactions)
            return False, len(reactions)
        
        # If not consecutive, just need both present
        return True, len(reactions)
    
    def detect_nitrile_to_amidoxime(self, rxn):
        """Detect nitrile to amidoxime conversion"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        # Check for nitrile in reactants and amidoxime in products
        nitrile_pattern = Chem.MolFromSmarts("[C]#[N]")  # Nitrile group
        amidoxime_pattern = Chem.MolFromSmarts("[C](=[N][O])[N]")  # Amidoxime group
        
        # Parse reactants and products
        try:
            reactant_mols = [Chem.MolFromSmiles(smi.strip()) for smi in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(smi.strip()) for smi in products.split(".")]
            
            # Remove None molecules
            reactant_mols = [mol for mol in reactant_mols if mol is not None]
            product_mols = [mol for mol in product_mols if mol is not None]
            
            # Check for nitrile in reactants
            has_nitrile = any(mol.HasSubstructMatch(nitrile_pattern) for mol in reactant_mols)
            
            # Check for amidoxime in products
            has_amidoxime = any(mol.HasSubstructMatch(amidoxime_pattern) for mol in product_mols)
            
            return has_nitrile and has_amidoxime
            
        except:
            return False
    
    def detect_amidoxime_to_oxadiazolone(self, rxn):
        """Detect amidoxime to oxadiazolone conversion"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        # Check for amidoxime in reactants and oxadiazolone in products
        amidoxime_pattern = Chem.MolFromSmarts("[C](=[N][O])[N]")  # Amidoxime group
        oxadiazolone_pattern = Chem.MolFromSmarts("[C]1=[N][O][C](=O)[N]1")  # 1,2,4-oxadiazol-3(2H)-one
        
        # Parse reactants and products
        try:
            reactant_mols = [Chem.MolFromSmiles(smi.strip()) for smi in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(smi.strip()) for smi in products.split(".")]
            
            # Remove None molecules
            reactant_mols = [mol for mol in reactant_mols if mol is not None]
            product_mols = [mol for mol in product_mols if mol is not None]
            
            # Check for amidoxime in reactants
            has_amidoxime = any(mol.HasSubstructMatch(amidoxime_pattern) for mol in reactant_mols)
            
            # Check for oxadiazolone in products
            has_oxadiazolone = any(mol.HasSubstructMatch(oxadiazolone_pattern) for mol in product_mols)
            
            return has_amidoxime and has_oxadiazolone
            
        except:
            return False
