"""Generated evaluation code for: Sequential ester protection and deprotection cycling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SequentialEsterCycling(MultiRxnCondBase):
    """
    Evaluates routes for sequential ester protection and deprotection cycling.
    Detects when ester groups are removed and then reinstalled later in the synthesis,
    specifically looking for patterns like Krapcho decarboxylation followed by 
    nitrile-to-ester conversion.
    """
    
    def __init__(self, config):
        self.protection_type = config.get("protection_type", "ester")
        self.cycle_count = config.get("cycle_count", 2)
        self.involves_same_functional_group = config.get("involves_same_functional_group", True)
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Track ester removal and installation events
        ester_events = []
        
        for i, rxn in enumerate(reactions):
            if self.detect_ester_removal(rxn):
                ester_events.append(('removal', i))
            elif self.detect_ester_installation(rxn):
                ester_events.append(('installation', i))
        
        # Check for cycling pattern
        has_cycle = self.check_cycling_pattern(ester_events)
        
        return has_cycle, len(reactions)
    
    def detect_ester_removal(self, rxn):
        """Detect ester removal reactions (e.g., Krapcho decarboxylation)"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = Chem.MolFromSmiles(rxn_parts[0])
        products_smiles = rxn_parts[1].split(".")
        
        if not reactants:
            return False
            
        # Common ester patterns
        ester_patterns = [
            "C(=O)OC",  # Methyl ester
            "C(=O)OCC",  # Ethyl ester
            "C(=O)O[CH3]",  # Methyl ester (explicit)
            "C(=O)OC(C)(C)C"  # tert-Butyl ester
        ]
        
        # Check if reactant has ester
        has_ester_reactant = False
        for pattern in ester_patterns:
            ester_mol = Chem.MolFromSmarts(pattern)
            if ester_mol and reactants.HasSubstructMatch(ester_mol):
                has_ester_reactant = True
                break
        
        if not has_ester_reactant:
            return False
            
        # Check if products have fewer or no esters
        for prod_smiles in products_smiles:
            prod_mol = Chem.MolFromSmiles(prod_smiles)
            if prod_mol:
                for pattern in ester_patterns:
                    ester_mol = Chem.MolFromSmarts(pattern)
                    if ester_mol and not prod_mol.HasSubstructMatch(ester_mol):
                        # Also check for characteristic Krapcho pattern (loss of CO2 + alcohol)
                        if self.detect_krapcho_pattern(rxn):
                            return True
        
        return False
    
    def detect_ester_installation(self, rxn):
        """Detect ester installation reactions"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants_smiles = rxn_parts[0].split(".")
        products = Chem.MolFromSmiles(rxn_parts[1])
        
        if not products:
            return False
            
        # Common ester patterns
        ester_patterns = [
            "C(=O)OC",  # Methyl ester
            "C(=O)OCC",  # Ethyl ester
            "C(=O)O[CH3]",  # Methyl ester (explicit)
            "C(=O)OC(C)(C)C"  # tert-Butyl ester
        ]
        
        # Check if product has ester
        has_ester_product = False
        for pattern in ester_patterns:
            ester_mol = Chem.MolFromSmarts(pattern)
            if ester_mol and products.HasSubstructMatch(ester_mol):
                has_ester_product = True
                break
        
        if not has_ester_product:
            return False
            
        # Check if reactants lack the ester (indicating installation)
        for react_smiles in reactants_smiles:
            react_mol = Chem.MolFromSmiles(react_smiles)
            if react_mol:
                # Look for nitrile to ester conversion
                nitrile_pattern = Chem.MolFromSmarts("C#N")
                if nitrile_pattern and react_mol.HasSubstructMatch(nitrile_pattern):
                    return True
                    
                # Look for carboxylic acid to ester conversion
                acid_pattern = Chem.MolFromSmarts("C(=O)O")
                if acid_pattern and react_mol.HasSubstructMatch(acid_pattern):
                    return True
        
        return False
    
    def detect_krapcho_pattern(self, rxn):
        """Detect characteristic Krapcho decarboxylation pattern"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = Chem.MolFromSmiles(rxn_parts[0])
        products_smiles = rxn_parts[1].split(".")
        
        if not reactants:
            return False
            
        # Look for malonate-type ester pattern in reactants
        malonate_pattern = Chem.MolFromSmarts("C(C(=O)OC)C(=O)OC")
        if malonate_pattern and reactants.HasSubstructMatch(malonate_pattern):
            # Check if products contain fewer ester groups
            for prod_smiles in products_smiles:
                if "C(=O)O" in prod_smiles or "C#N" in prod_smiles:
                    return True
        
        return False
    
    def check_cycling_pattern(self, ester_events):
        """Check if ester events form a cycling pattern"""
        if len(ester_events) < self.cycle_count:
            return False
            
        # Look for alternating removal/installation pattern
        removal_steps = [step for event, step in ester_events if event == 'removal']
        installation_steps = [step for event, step in ester_events if event == 'installation']
        
        # Must have at least one removal followed by installation
        if not removal_steps or not installation_steps:
            return False
            
        # Check if removal happens before installation (cycling pattern)
        min_removal = min(removal_steps)
        max_installation = max(installation_steps)
        
        return min_removal < max_installation and len(removal_steps) >= 1 and len(installation_steps) >= 1
