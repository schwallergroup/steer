"""Generated evaluation code for: Nitrile as temporary functional group intermediate"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class NitrileTemporaryIntermediate(MultiRxnCondBase):
    """
    Evaluates synthesis routes for the use of nitrile as a temporary functional group.
    Checks if amide is converted to nitrile, persists for multiple steps, then converted to ester.
    """
    
    def __init__(self, config):
        self.min_intermediate_steps = config.get("intermediate_steps", 3)
        self.final_conversion = config.get("final_conversion", "nitrile_to_ester")
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Find amide to nitrile conversion
        amide_to_nitrile_idx = -1
        for i, rxn in enumerate(reactions):
            if self.detect_amide_to_nitrile(rxn):
                amide_to_nitrile_idx = i
                break
        
        if amide_to_nitrile_idx == -1:
            return False, len(reactions)
        
        # Find nitrile to ester conversion after sufficient intermediate steps
        for i in range(amide_to_nitrile_idx + self.min_intermediate_steps, len(reactions)):
            if self.detect_nitrile_to_ester(reactions[i]):
                # Check that nitrile persists in intermediate steps
                nitrile_persists = True
                for j in range(amide_to_nitrile_idx + 1, i):
                    if not self.has_nitrile_intermediate(reactions[j]):
                        nitrile_persists = False
                        break
                
                if nitrile_persists:
                    return True, len(reactions)
        
        return False, len(reactions)
    
    def detect_amide_to_nitrile(self, rxn):
        """Detect conversion of amide to nitrile"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[0].split(".") if r.strip()]
        products = [Chem.MolFromSmiles(p.strip()) for p in rxn_parts[1].split(".") if p.strip()]
        
        if not all(reactants) or not all(products):
            return False
        
        # Look for amide pattern in reactants
        amide_pattern = Chem.MolFromSmarts("[C](=[O])[NH]")
        has_amide_reactant = any(mol.HasSubstructMatch(amide_pattern) for mol in reactants)
        
        # Look for nitrile pattern in products
        nitrile_pattern = Chem.MolFromSmarts("[C]#[N]")
        has_nitrile_product = any(mol.HasSubstructMatch(nitrile_pattern) for mol in products)
        
        return has_amide_reactant and has_nitrile_product
    
    def detect_nitrile_to_ester(self, rxn):
        """Detect conversion of nitrile to ester"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[0].split(".") if r.strip()]
        products = [Chem.MolFromSmiles(p.strip()) for p in rxn_parts[1].split(".") if p.strip()]
        
        if not all(reactants) or not all(products):
            return False
        
        # Look for nitrile pattern in reactants
        nitrile_pattern = Chem.MolFromSmarts("[C]#[N]")
        has_nitrile_reactant = any(mol.HasSubstructMatch(nitrile_pattern) for mol in reactants)
        
        # Look for ester pattern in products
        ester_pattern = Chem.MolFromSmarts("[C](=[O])[O][C]")
        has_ester_product = any(mol.HasSubstructMatch(ester_pattern) for mol in products)
        
        return has_nitrile_reactant and has_ester_product
    
    def has_nitrile_intermediate(self, rxn):
        """Check if reaction involves nitrile as intermediate (present in both reactants and products)"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[0].split(".") if r.strip()]
        products = [Chem.MolFromSmiles(p.strip()) for p in rxn_parts[1].split(".") if p.strip()]
        
        if not all(reactants) or not all(products):
            return False
        
        nitrile_pattern = Chem.MolFromSmarts("[C]#[N]")
        has_nitrile_reactant = any(mol.HasSubstructMatch(nitrile_pattern) for mol in reactants)
        has_nitrile_product = any(mol.HasSubstructMatch(nitrile_pattern) for mol in products)
        
        return has_nitrile_reactant or has_nitrile_product
