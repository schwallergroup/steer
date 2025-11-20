"""Generated evaluation code for: Methylthio protecting group on imidazole nitrogen"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class MethylthioImidazoleProtection(MultiRxnCondBase):
    """
    Evaluates synthesis routes for methylthio protecting group strategy on imidazole nitrogen.
    Checks for installation at specified depth and removal at specified depth.
    """
    
    def __init__(self, config):
        self.installation_step = config["parameters"]["installation_step"]
        self.removal_step = config["parameters"]["removal_step"]
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        total_steps = len(reactions)
        
        # Check for installation at specified depth from start
        installation_found = False
        installation_depth = -1
        for i, rxn in enumerate(reactions):
            if self.detect_methylthio_installation(rxn):
                installation_found = True
                installation_depth = i + 1
                break
                
        # Check for removal at specified depth from end
        removal_found = False
        removal_depth = -1
        for i, rxn in enumerate(reversed(reactions)):
            if self.detect_methylthio_removal(rxn):
                removal_found = True
                removal_depth = i + 1
                break
        
        # Check if installation and removal occur at target depths
        installation_correct = (installation_found and 
                              installation_depth == self.installation_step)
        removal_correct = (removal_found and 
                          removal_depth == self.removal_step)
        
        condition_met = installation_correct and removal_correct
        
        return condition_met, total_steps
    
    def detect_methylthio_installation(self, rxn):
        """Detect installation of methylthio group on imidazole nitrogen"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = [Chem.MolFromSmiles(r) for r in rxn_parts[0].split(".")]
        products = [Chem.MolFromSmiles(p) for p in rxn_parts[1].split(".")]
        
        if not all(reactants) or not all(products):
            return False
        
        # Pattern for imidazole
        imidazole_pattern = Chem.MolFromSmarts("[nH]1c[nH]cc1")
        # Pattern for N-methylthio imidazole
        methylthio_imidazole_pattern = Chem.MolFromSmarts("n1c[nH]cc1SC")
        
        # Check if reactant has imidazole and product has methylthio-imidazole
        has_imidazole_reactant = any(mol.HasSubstructMatch(imidazole_pattern) 
                                   for mol in reactants)
        has_methylthio_product = any(mol.HasSubstructMatch(methylthio_imidazole_pattern) 
                                   for mol in products)
        
        # Also check for methylthio reagent (like methylthiol or dimethyl disulfide)
        methylthio_reagent_patterns = [
            Chem.MolFromSmarts("CSC"),  # dimethyl disulfide
            Chem.MolFromSmarts("CS"),   # methylthiol
            Chem.MolFromSmarts("CSCl")  # methylsulfenyl chloride
        ]
        
        has_methylthio_reagent = any(
            any(mol.HasSubstructMatch(pattern) for pattern in methylthio_reagent_patterns)
            for mol in reactants
        )
        
        return has_imidazole_reactant and has_methylthio_product and has_methylthio_reagent
    
    def detect_methylthio_removal(self, rxn):
        """Detect removal of methylthio group (desulfurization)"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = [Chem.MolFromSmiles(r) for r in rxn_parts[0].split(".")]
        products = [Chem.MolFromSmiles(p) for p in rxn_parts[1].split(".")]
        
        if not all(reactants) or not all(products):
            return False
        
        # Pattern for N-methylthio imidazole
        methylthio_imidazole_pattern = Chem.MolFromSmarts("n1c[nH]cc1SC")
        # Pattern for imidazole after deprotection
        imidazole_pattern = Chem.MolFromSmarts("[nH]1c[nH]cc1")
        
        # Check if reactant has methylthio-imidazole and product has free imidazole
        has_methylthio_reactant = any(mol.HasSubstructMatch(methylthio_imidazole_pattern) 
                                    for mol in reactants)
        has_imidazole_product = any(mol.HasSubstructMatch(imidazole_pattern) 
                                  for mol in products)
        
        # Check for desulfurization reagents (Raney Ni, Pd/C, etc.)
        desulfurization_reagents = [
            Chem.MolFromSmarts("[Ni]"),  # Raney nickel
            Chem.MolFromSmarts("[Pd]"),  # Palladium
            Chem.MolFromSmarts("P(c1ccccc1)(c2ccccc2)c3ccccc3")  # PPh3
        ]
        
        has_desulfurization_reagent = any(
            any(mol.HasSubstructMatch(pattern) for pattern in desulfurization_reagents)
            for mol in reactants
        )
        
        return has_methylthio_reactant and has_imidazole_product and has_desulfurization_reagent
