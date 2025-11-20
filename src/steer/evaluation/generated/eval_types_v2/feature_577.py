"""Generated evaluation code for: Three step carbonyl installation via ester"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ThreeStepCarbonylInstallation(MultiRxnCondBase):
    """
    Evaluates routes that use a three-step carbonyl installation via ester:
    1. Carbonylation to form ester
    2. Hydrolysis of ester to carboxylic acid
    3. Formation of Weinreb amide from carboxylic acid
    
    Checks for presence of this specific sequence and formation of Weinreb amide.
    """
    
    def __init__(self, config):
        self.target_sequence = config["parameters"]["sequence"]  # ["carbonylation", "hydrolysis", "amide_formation"]
        self.target_functional_group = config["parameters"]["target_functional_group"]  # "weinreb_amide"
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Check for Weinreb amide formation
        has_weinreb_amide = any(self.detect_weinreb_amide_formation(r) for r in reactions)
        
        # Check for the three-step sequence
        has_sequence = self.detect_three_step_sequence(reactions)
        
        condition = has_weinreb_amide and has_sequence
        return condition, len(reactions)
    
    def detect_weinreb_amide_formation(self, rxn):
        """Detect formation of Weinreb amide (N-methoxy-N-methyl amide)"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        # Weinreb amide pattern: N(C)(OC)C=O
        weinreb_pattern = Chem.MolFromSmarts("[N;X3]([CH3])([O][CH3])[C;X3](=[O])")
        
        try:
            # Check if Weinreb amide is formed in products
            for prod_smiles in products.split("."):
                prod_mol = Chem.MolFromSmiles(prod_smiles)
                if prod_mol and prod_mol.HasSubstructMatch(weinreb_pattern):
                    # Check if Weinreb amide was not present in reactants
                    weinreb_in_reactants = False
                    for react_smiles in reactants.split("."):
                        react_mol = Chem.MolFromSmiles(react_smiles)
                        if react_mol and react_mol.HasSubstructMatch(weinreb_pattern):
                            weinreb_in_reactants = True
                            break
                    
                    if not weinreb_in_reactants:
                        return True
        except:
            return False
            
        return False
    
    def detect_three_step_sequence(self, reactions):
        """Detect the three-step sequence: carbonylation -> hydrolysis -> amide formation"""
        if len(reactions) < 3:
            return False
            
        # Track presence of each step
        has_carbonylation = any(self.detect_carbonylation(r) for r in reactions)
        has_hydrolysis = any(self.detect_ester_hydrolysis(r) for r in reactions)
        has_amide_formation = any(self.detect_amide_formation(r) for r in reactions)
        
        return has_carbonylation and has_hydrolysis and has_amide_formation
    
    def detect_carbonylation(self, rxn):
        """Detect carbonylation reaction (CO insertion to form C=O)"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        # Look for CO as reactant and ester/acid formation
        co_pattern = "[C-]#[O+]"  # Carbon monoxide
        ester_pattern = "[C;X3](=[O])[O][C;!$(C=O)]"  # Ester pattern
        
        try:
            has_co = any("C#O" in r or Chem.MolFromSmiles(r.strip()) and 
                        Chem.MolFromSmiles(r.strip()).HasSubstructMatch(Chem.MolFromSmarts(co_pattern))
                        for r in reactants.split("."))
            
            has_ester_product = any(Chem.MolFromSmiles(p.strip()) and
                                  Chem.MolFromSmiles(p.strip()).HasSubstructMatch(Chem.MolFromSmarts(ester_pattern))
                                  for p in products.split("."))
            
            return has_co or has_ester_product
        except:
            return False
    
    def detect_ester_hydrolysis(self, rxn):
        """Detect hydrolysis of ester to carboxylic acid"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        ester_pattern = "[C;X3](=[O])[O][C;!$(C=O)]"  # Ester
        acid_pattern = "[C;X3](=[O])[OH]"  # Carboxylic acid
        
        try:
            has_ester_reactant = any(Chem.MolFromSmiles(r.strip()) and
                                   Chem.MolFromSmiles(r.strip()).HasSubstructMatch(Chem.MolFromSmarts(ester_pattern))
                                   for r in reactants.split("."))
            
            has_acid_product = any(Chem.MolFromSmiles(p.strip()) and
                                 Chem.MolFromSmiles(p.strip()).HasSubstructMatch(Chem.MolFromSmarts(acid_pattern))
                                 for p in products.split("."))
            
            return has_ester_reactant and has_acid_product
        except:
            return False
    
    def detect_amide_formation(self, rxn):
        """Detect amide formation from carboxylic acid"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        acid_pattern = "[C;X3](=[O])[OH]"  # Carboxylic acid
        amide_pattern = "[C;X3](=[O])[N]"  # Amide
        
        try:
            has_acid_reactant = any(Chem.MolFromSmiles(r.strip()) and
                                  Chem.MolFromSmiles(r.strip()).HasSubstructMatch(Chem.MolFromSmarts(acid_pattern))
                                  for r in reactants.split("."))
            
            has_amide_product = any(Chem.MolFromSmiles(p.strip()) and
                                  Chem.MolFromSmiles(p.strip()).HasSubstructMatch(Chem.MolFromSmarts(amide_pattern))
                                  for p in products.split("."))
            
            return has_acid_reactant and has_amide_product
        except:
            return False
