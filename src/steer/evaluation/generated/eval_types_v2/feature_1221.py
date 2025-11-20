"""Generated evaluation code for: Late imidazopyrazine ring formation via cyclization"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ImidazopyrazineFormation(BaseScoring):
    """
    Evaluates synthesis routes for late-stage imidazopyrazine ring formation via cyclization.
    Detects when the imidazopyrazine core (n1ccnc2nccc12) is formed through intramolecular
    cyclization rather than building from pre-formed heterocycles.
    """
    
    def __init__(self, config):
        self.ring_smarts = config["parameters"]["ring_smarts"]  # "n1ccnc2nccc12"
        self.timing = config["parameters"]["timing"]  # "late"
        self.formation_method = config["parameters"]["formation_method"]  # "intramolecular_cyclization"
        
    def route_scoring(self, x):
        if x < 0:
            return 0  # Ring formation doesn't happen via cyclization
        else:
            # Late-stage formation is better - higher score for later depths
            return 1 - x  # x is depth fraction, so 1-x rewards later formation
            
    def hit_condition(self, d):
        """Check if this reaction forms imidazopyrazine via intramolecular cyclization"""
        try:
            rxn = d["metadata"]["mapped_reaction_smiles"].split(">>")
            products = rxn[0]
            reactants = rxn[1].split(".")
            
            # Check if product contains imidazopyrazine ring
            prod_mol = Chem.MolFromSmiles(products)
            if not prod_mol:
                return False
                
            imidazopyrazine_pattern = Chem.MolFromSmarts(self.ring_smarts)
            if not prod_mol.HasSubstructMatch(imidazopyrazine_pattern):
                return False
            
            # Check if any reactant already contains the complete imidazopyrazine core
            for reactant_smiles in reactants:
                reactant_mol = Chem.MolFromSmiles(reactant_smiles)
                if reactant_mol and reactant_mol.HasSubstructMatch(imidazopyrazine_pattern):
                    return False  # Ring already exists, not formation
            
            # Check if this is intramolecular cyclization by verifying:
            # 1. Single reactant case (intramolecular)
            # 2. Or reactants contain pieces that form the ring system
            if len(reactants) == 1:
                # Single reactant forming ring - definitely intramolecular
                return True
            elif len(reactants) == 2:
                # Check if reactants contain complementary N-heterocyclic fragments
                # that could cyclize to form imidazopyrazine
                reactant_mols = [Chem.MolFromSmiles(r) for r in reactants if Chem.MolFromSmiles(r)]
                
                # Look for pyrazine-like and imidazole-like fragments
                pyrazine_like = Chem.MolFromSmarts("nccn")  # pyrazine fragment
                imidazole_like = Chem.MolFromSmarts("[nH]ccn")  # imidazole-like fragment
                
                has_pyrazine_fragment = any(mol.HasSubstructMatch(pyrazine_like) for mol in reactant_mols)
                has_imidazole_fragment = any(mol.HasSubstructMatch(imidazole_like) for mol in reactant_mols)
                
                return has_pyrazine_fragment or has_imidazole_fragment
            
            return False
            
        except (KeyError, AttributeError, ValueError):
            return False
