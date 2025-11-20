"""Generated evaluation code for: Early pyrimidine core construction via condensation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlyPyrimidineCondensation(BaseScoring):
    """
    Evaluates whether pyrimidine core construction occurs early in the synthesis
    via condensation between guanidine and dicarbonyl substrates.
    
    Returns higher scores when pyrimidine formation happens in early stages
    of the synthetic route through the specified condensation mechanism.
    """
    
    def __init__(self, config: Dict):
        self.timing = config["parameters"]["timing"]
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Pyrimidine condensation doesn't happen
        else:
            # Early stage formation gets higher score
            # x is depth fraction (0 = start, 1 = end)
            return 1 - x
    
    def hit_condition(self, d):
        """Check if this reaction forms pyrimidine via guanidine-dicarbonyl condensation"""
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            rxn_parts = rxn_smiles.split(">>")
            
            if len(rxn_parts) != 2:
                return False
                
            reactants = rxn_parts[0].split(".")
            product = rxn_parts[1]
            
            # Check if product contains pyrimidine core
            product_mol = Chem.MolFromSmiles(product)
            if not product_mol:
                return False
                
            pyrimidine_pattern = Chem.MolFromSmarts("n1cnccc1")
            if not product_mol.HasSubstructMatch(pyrimidine_pattern):
                return False
            
            # Check if reactants contain guanidine and dicarbonyl patterns
            has_guanidine = False
            has_dicarbonyl = False
            
            guanidine_pattern = Chem.MolFromSmarts("NC(=N)N")  # guanidine core
            dicarbonyl_patterns = [
                Chem.MolFromSmarts("C(=O)CC(=O)"),  # 1,3-dicarbonyl
                Chem.MolFromSmarts("C(=O)C(=O)"),   # 1,2-dicarbonyl
                Chem.MolFromSmarts("O=C-C-C=O")     # general dicarbonyl
            ]
            
            for reactant_smiles in reactants:
                reactant_mol = Chem.MolFromSmiles(reactant_smiles)
                if not reactant_mol:
                    continue
                    
                # Check for guanidine substructure
                if reactant_mol.HasSubstructMatch(guanidine_pattern):
                    has_guanidine = True
                    
                # Check for dicarbonyl substructures
                for pattern in dicarbonyl_patterns:
                    if reactant_mol.HasSubstructMatch(pattern):
                        has_dicarbonyl = True
                        break
            
            # Return True only if we have both required substrates and pyrimidine formation
            return has_guanidine and has_dicarbonyl
            
        except (KeyError, AttributeError, ValueError):
            return False
