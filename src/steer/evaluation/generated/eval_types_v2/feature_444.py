"""Generated evaluation code for: Convergent synthesis via Suzuki coupling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentSuzukiCoupling(BaseScoring):
    """
    Evaluates convergent synthesis routes that use Suzuki coupling to join fragments.
    Checks for Suzuki-Miyaura cross-coupling reactions that combine two substantial
    molecular fragments at the specified position in the route.
    """
    
    def __init__(self, config: Dict):
        self.fragments_joined = config["parameters"].get("fragments_joined", 2)
        self.step_position = config["parameters"].get("step_position", "middle")
        
        # Suzuki coupling pattern: formation of C-C bond between aryl/vinyl groups
        self.suzuki_product_pattern = "[c,C]([!#1])([!#1])-[c,C]([!#1])([!#1])"
        
        # Boronic acid/ester patterns for reactants
        self.boronic_patterns = [
            "[c,C]-B(O)O",  # Boronic acid
            "[c,C]-B1OC(C)(C)C(C)(C)O1",  # Pinacol boronate
            "[c,C]-B1OCCO1"  # Ethylene glycol boronate
        ]
        
        # Halide patterns for coupling partner
        self.halide_pattern = "[c,C][Br,I,Cl]"
    
    def route_scoring(self, x: float) -> float:
        """Score based on when Suzuki coupling occurs in the route."""
        if x < 0:
            return 0  # Reaction doesn't occur
            
        # For convergent synthesis, middle positions are preferred
        if self.step_position == "middle":
            # Optimal around 0.3-0.7 depth (middle of route)
            if 0.3 <= x <= 0.7:
                return 1.0
            elif 0.1 <= x <= 0.9:
                return 0.7
            else:
                return 0.3
        elif self.step_position == "early":
            return 1.0 - x  # Earlier is better
        elif self.step_position == "late":
            return x  # Later is better
        else:
            return 0.5  # Default moderate score
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction is a Suzuki coupling joining substantial fragments."""
        metadata = d.get("metadata", {})
        
        # Check if reaction smiles is available
        mapped_rxn = metadata.get("mapped_reaction_smiles")
        if not mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            product_smiles = rxn_parts[0]
            reactant_smiles = rxn_parts[1]
            
            product_mol = Chem.MolFromSmiles(product_smiles)
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactant_smiles.split(".")]
            
            if not product_mol or len(reactant_mols) < 2:
                return False
                
            # Check if product has the characteristic C-C bond formation pattern
            suzuki_pattern_mol = Chem.MolFromSmarts(self.suzuki_product_pattern)
            if not product_mol.HasSubstructMatch(suzuki_pattern_mol):
                return False
                
            # Check for Suzuki coupling reactants
            has_boronic = False
            has_halide = False
            substantial_fragments = 0
            
            for reactant in reactant_mols:
                if not reactant:
                    continue
                    
                # Check for boronic acid/ester
                for boronic_pattern in self.boronic_patterns:
                    boronic_mol = Chem.MolFromSmarts(boronic_pattern)
                    if reactant.HasSubstructMatch(boronic_mol):
                        has_boronic = True
                        if reactant.GetNumHeavyAtoms() >= 6:  # Substantial fragment
                            substantial_fragments += 1
                        break
                
                # Check for halide
                halide_mol = Chem.MolFromSmarts(self.halide_pattern)
                if reactant.HasSubstructMatch(halide_mol):
                    has_halide = True
                    if reactant.GetNumHeavyAtoms() >= 6:  # Substantial fragment
                        substantial_fragments += 1
            
            # Must have both Suzuki components and join substantial fragments
            return (has_boronic and has_halide and 
                    substantial_fragments >= self.fragments_joined)
                    
        except Exception:
            return False
