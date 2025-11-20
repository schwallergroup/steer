"""Generated evaluation code for: Early stage aryl chloride hydrolysis"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ArylChlorideHydrolysis(BaseScoring):
    """
    Evaluates early stage aryl chloride hydrolysis reactions.
    Detects C-Cl bond breaking in aromatic systems and penalizes late-stage occurrences.
    """
    
    def __init__(self, config: Dict):
        self.bond_smarts = config["parameters"]["bond_smarts"]  # "[cH0:1][Cl:2]"
        self.depth_threshold = config["parameters"]["depth_threshold"]  # 7
        self.timing = config["parameters"]["timing"]  # "early"
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Bond break doesn't happen
        
        if self.timing == "early":
            # Penalize late-stage aryl chloride hydrolysis
            if x > (self.depth_threshold / 10.0):  # Convert depth to fraction
                return 10  # High penalty for late-stage
            else:
                return 0   # Good - early stage
        else:
            # If timing were "late", reward late-stage
            return 10 * (1 - x)
    
    def hit_condition(self, d):
        """Check if this reaction breaks an aryl C-Cl bond"""
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
                
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse molecules
            reactants = [Chem.MolFromSmiles(smi) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi) for smi in products_smiles.split(".")]
            
            if not all(reactants) or not all(products):
                return False
            
            # Create pattern for aryl chloride
            pattern = Chem.MolFromSmarts(self.bond_smarts)
            if not pattern:
                return False
            
            # Check if aryl chloride is present in reactants but not products
            reactant_has_pattern = any(mol.HasSubstructMatch(pattern) for mol in reactants)
            product_has_pattern = any(mol.HasSubstructMatch(pattern) for mol in products)
            
            # Check for atom mapping to confirm bond breaking
            if reactant_has_pattern and not product_has_pattern:
                return self._confirm_bond_break_by_mapping(reactants_smiles, products_smiles)
            
            return False
            
        except Exception:
            return False
    
    def _confirm_bond_break_by_mapping(self, reactants_smiles, products_smiles):
        """Use atom mapping to confirm the specific C-Cl bond is broken"""
        try:
            reactants = [Chem.MolFromSmiles(smi) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi) for smi in products_smiles.split(".")]
            
            # Find mapped aryl carbon and chlorine atoms
            aryl_carbon_map = None
            chlorine_map = None
            
            for mol in reactants:
                matches = mol.GetSubstructMatches(Chem.MolFromSmarts(self.bond_smarts))
                for match in matches:
                    c_idx, cl_idx = match
                    aryl_carbon_map = mol.GetAtomWithIdx(c_idx).GetAtomMapNum()
                    chlorine_map = mol.GetAtomWithIdx(cl_idx).GetAtomMapNum()
                    break
                if aryl_carbon_map and chlorine_map:
                    break
            
            if not (aryl_carbon_map and chlorine_map):
                return False
            
            # Check if these atoms are in different product molecules (bond broken)
            carbon_product = None
            chlorine_product = None
            
            for mol in products:
                for atom in mol.GetAtoms():
                    if atom.GetAtomMapNum() == aryl_carbon_map:
                        carbon_product = mol
                    elif atom.GetAtomMapNum() == chlorine_map:
                        chlorine_product = mol
            
            # Bond is broken if atoms are in different product molecules
            return carbon_product is not chlorine_product if (carbon_product and chlorine_product) else False
            
        except Exception:
            return False
