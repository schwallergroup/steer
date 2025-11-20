"""Generated evaluation code for: Multi-step decarboxylation via amide-nitrile-acid sequence"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class MultiStepDecarboxylation(MultiRxnCondBase):
    """
    Detects multi-step decarboxylation routes that go through amide-nitrile-acid-ester intermediates
    instead of using direct decarboxylation methods.
    """
    
    def __init__(self, config):
        self.required_intermediates = config["parameters"].get("intermediate_steps", ["amide", "nitrile", "acid", "ester"])
        self.bond_type = config["parameters"].get("bond_type", "C-COOH")
        self.method = config["parameters"].get("method", "indirect")
        
        # SMARTS patterns for intermediate detection
        self.patterns = {
            "amide": "[C](=[O])[N]",
            "nitrile": "[C]#[N]",
            "acid": "[C](=[O])[OH]",
            "ester": "[C](=[O])[O][C]",
            "carboxyl": "[C](=[O])[OH]"
        }
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Check if we have carboxyl group removal (decarboxylation)
        has_decarboxylation = any(self.detect_decarboxylation(r) for r in reactions)
        
        if not has_decarboxylation:
            return False, len(reactions)
        
        # Check for presence of required intermediates
        intermediates_found = set()
        for rxn in reactions:
            for intermediate_type in self.required_intermediates:
                if self.detect_intermediate_formation(rxn, intermediate_type):
                    intermediates_found.add(intermediate_type)
        
        # Check if we found the required sequence
        required_set = set(self.required_intermediates)
        has_indirect_sequence = len(intermediates_found.intersection(required_set)) >= 3
        
        # Also check that no direct decarboxylation is used
        has_direct_decarb = any(self.detect_direct_decarboxylation(r) for r in reactions)
        
        condition_met = has_decarboxylation and has_indirect_sequence and not has_direct_decarb
        
        return condition_met, len(reactions)
    
    def detect_decarboxylation(self, rxn):
        """Detect if CO2 is lost in the reaction (decarboxylation)"""
        reactants_smiles, products_smiles = rxn.split(">>")
        
        # Count carboxyl groups in reactants vs products
        reactants = [Chem.MolFromSmiles(smi) for smi in reactants_smiles.split(".") if smi.strip()]
        products = [Chem.MolFromSmiles(smi) for smi in products_smiles.split(".") if smi.strip()]
        
        carboxyl_pattern = Chem.MolFromSmarts(self.patterns["carboxyl"])
        
        reactant_carboxyls = sum(len(mol.GetSubstructMatches(carboxyl_pattern)) 
                                for mol in reactants if mol is not None)
        product_carboxyls = sum(len(mol.GetSubstructMatches(carboxyl_pattern)) 
                               for mol in products if mol is not None)
        
        # Also check for CO2 as a product
        co2_present = "O=C=O" in products_smiles or "[O-]C(=O)[O-]" in products_smiles
        
        return (reactant_carboxyls > product_carboxyls) or co2_present
    
    def detect_direct_decarboxylation(self, rxn):
        """Detect direct decarboxylation reactions (thermal, enzymatic, etc.)"""
        reactants_smiles, products_smiles = rxn.split(">>")
        
        # Simple heuristic: direct decarboxylation usually produces CO2 directly
        # and doesn't involve complex intermediates in the same step
        has_co2 = "O=C=O" in products_smiles
        
        # Check if reactants contain complex intermediates
        complex_intermediates = any(intermediate in reactants_smiles.lower() 
                                  for intermediate in ["amide", "nitrile"])
        
        return has_co2 and not complex_intermediates
    
    def detect_intermediate_formation(self, rxn, intermediate_type):
        """Detect formation of specific intermediate types"""
        if intermediate_type not in self.patterns:
            return False
            
        reactants_smiles, products_smiles = rxn.split(">>")
        
        pattern = Chem.MolFromSmarts(self.patterns[intermediate_type])
        if pattern is None:
            return False
        
        # Check if intermediate is formed (present in products but not reactants, or more in products)
        reactants = [Chem.MolFromSmiles(smi) for smi in reactants_smiles.split(".") if smi.strip()]
        products = [Chem.MolFromSmiles(smi) for smi in products_smiles.split(".") if smi.strip()]
        
        reactant_matches = sum(len(mol.GetSubstructMatches(pattern)) 
                              for mol in reactants if mol is not None)
        product_matches = sum(len(mol.GetSubstructMatches(pattern)) 
                             for mol in products if mol is not None)
        
        return product_matches > reactant_matches
    
    def route_scoring(self, x):
        """Score based on route length - longer indirect routes score higher (worse)"""
        if x < 0:
            return 0  # Condition not met
        else:
            # Penalize longer routes more heavily as they represent more convoluted synthesis
            return min(10, x * 2)  # Scale route length, cap at 10
