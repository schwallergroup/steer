"""Generated evaluation code for: Late stage Suzuki coupling for pyridine installation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageSuzukiPyridine(BaseScoring):
    """
    Evaluates whether a late-stage Suzuki coupling occurs for pyridine installation.
    Checks for formation of pyridine-thiophene bonds via Suzuki coupling reactions.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "fraction")
        self.target_depth = config.get("target_depth", {}).get("value", 0.8)
        
        # SMARTS patterns for detection
        self.pyridine_pattern = "c1ccncc1"  # Pyridine ring
        self.thiophene_pattern = "c1ccsc1"  # Thiophene ring
        self.pyridine_thiophene_bond = "[c:1]1ccncc1-[c:2]1ccsc1"  # Connected pyridine-thiophene
        
        # Boronic acid patterns for Suzuki detection
        self.boronic_acid_pattern = "[cH0,c]-B(O)O"
        self.chloro_pattern = "[cH0,c]-Cl"

    def route_scoring(self, x) -> float:
        """Convert depth fraction to score (0-10). Late stage (high x) is better."""
        if x < 0:
            return 0  # Reaction doesn't occur
        
        if self.condition_type == "bool":
            return 10 if x >= self.target_depth else 0
        else:
            # Reward late-stage occurrence
            if x >= self.target_depth:
                return 10
            else:
                return 10 * (x / self.target_depth)

    def hit_condition(self, d) -> bool:
        """Check if this reaction node represents a Suzuki coupling forming pyridine-thiophene bond."""
        metadata = d.get("metadata", {})
        
        # Check if it's a Suzuki coupling reaction
        if not self._is_suzuki_coupling(metadata):
            return False
            
        # Check if pyridine-thiophene bond is formed
        return self._forms_pyridine_thiophene_bond(metadata)

    def _is_suzuki_coupling(self, metadata) -> bool:
        """Detect if reaction is a Suzuki coupling based on substrates."""
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        if not mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants = rxn_parts[1].split(".")
            
            # Look for boronic acid and halide patterns
            has_boronic_acid = False
            has_halide = False
            
            for reactant_smiles in reactants:
                mol = Chem.MolFromSmiles(reactant_smiles)
                if mol is None:
                    continue
                    
                # Check for boronic acid
                boronic_mol = Chem.MolFromSmarts(self.boronic_acid_pattern)
                if boronic_mol and mol.HasSubstructMatch(boronic_mol):
                    has_boronic_acid = True
                    
                # Check for chloride
                chloro_mol = Chem.MolFromSmarts(self.chloro_pattern)
                if chloro_mol and mol.HasSubstructMatch(chloro_mol):
                    has_halide = True
                    
            return has_boronic_acid and has_halide
            
        except:
            return False

    def _forms_pyridine_thiophene_bond(self, metadata) -> bool:
        """Check if the reaction forms a pyridine-thiophene bond."""
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        if not mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            product_mol = Chem.MolFromSmiles(rxn_parts[0])
            reactant_mols = [Chem.MolFromSmiles(r) for r in rxn_parts[1].split(".")]
            
            if product_mol is None:
                return False
                
            # Check if product contains pyridine-thiophene connection
            pyridine_thiophene_mol = Chem.MolFromSmarts(self.pyridine_thiophene_bond)
            if not (pyridine_thiophene_mol and product_mol.HasSubstructMatch(pyridine_thiophene_mol)):
                return False
                
            # Verify reactants contain separate pyridine and thiophene fragments
            pyridine_mol = Chem.MolFromSmarts(self.pyridine_pattern)
            thiophene_mol = Chem.MolFromSmarts(self.thiophene_pattern)
            
            has_pyridine_reactant = False
            has_thiophene_reactant = False
            
            for reactant in reactant_mols:
                if reactant is None:
                    continue
                    
                if pyridine_mol and reactant.HasSubstructMatch(pyridine_mol):
                    has_pyridine_reactant = True
                    
                if thiophene_mol and reactant.HasSubstructMatch(thiophene_mol):
                    has_thiophene_reactant = True
                    
            return has_pyridine_reactant and has_thiophene_reactant
            
        except:
            return False
