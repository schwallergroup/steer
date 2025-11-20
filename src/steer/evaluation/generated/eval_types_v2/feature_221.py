"""Generated evaluation code for: Sequential Suzuki couplings for biaryl assembly"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SequentialSuzukiCouplings(MultiRxnCondBase):
    """
    Evaluates routes for sequential Suzuki coupling reactions for biaryl assembly.
    Checks if exactly 2 Suzuki couplings occur in early-to-mid stage of synthesis.
    """
    
    def __init__(self, config):
        self.target_count = config.get("count", 2)
        self.timing = config.get("timing", "early_and_mid")
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        total_reactions = len(reactions)
        
        # Find all Suzuki coupling reactions and their positions
        suzuki_positions = []
        for i, rxn in enumerate(reactions):
            if self.detect_suzuki_coupling(rxn):
                # Convert position to depth fraction (0 = root, 1 = leaves)
                depth_fraction = i / max(1, total_reactions - 1)
                suzuki_positions.append(depth_fraction)
        
        # Check if we have the target count
        has_correct_count = len(suzuki_positions) == self.target_count
        
        # Check timing constraint for early_and_mid
        correct_timing = True
        if self.timing == "early_and_mid" and suzuki_positions:
            # Early and mid means reactions should occur in first 2/3 of route
            correct_timing = all(pos <= 0.67 for pos in suzuki_positions)
            # Also check that they're not all at the very end (> 0.8)
            correct_timing = correct_timing and not all(pos > 0.8 for pos in suzuki_positions)
        
        condition_met = has_correct_count and correct_timing
        
        # Return average depth of Suzuki couplings for scoring
        avg_depth = sum(suzuki_positions) / len(suzuki_positions) if suzuki_positions else -1
        
        return condition_met, avg_depth if condition_met else -1
    
    def detect_suzuki_coupling(self, rxn):
        """
        Detect Suzuki coupling reaction by looking for:
        1. Boronic acid/ester reactant (B(OH)2 or B-ester)
        2. Aryl halide reactant (Ar-X where X = Br, I, Cl)
        3. Formation of new C-C bond between aromatics
        """
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants_smiles = rxn_parts[0]
            product_smiles = rxn_parts[1]
            
            # Check for boronic acid/ester patterns
            boronic_patterns = [
                "[#6]-B(-O)-O",  # Boronic acid
                "[#6]-B(O)O",    # Boronic acid alternative
                "[#6]-B1OC(C)(C)CO1",  # Pinacol ester
                "[#6]-B1OCCO1"   # Ethylene glycol ester
            ]
            
            # Check for aryl halide patterns
            halide_patterns = [
                "c-Br",  # Aryl bromide
                "c-I",   # Aryl iodide  
                "c-Cl"   # Aryl chloride
            ]
            
            has_boronic = False
            has_halide = False
            
            for reactant in reactants_smiles.split("."):
                mol = Chem.MolFromSmiles(reactant)
                if mol is None:
                    continue
                    
                # Check for boronic acid/ester
                for pattern in boronic_patterns:
                    if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                        has_boronic = True
                        break
                
                # Check for aryl halide
                for pattern in halide_patterns:
                    if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                        has_halide = True
                        break
            
            # Additional check: look for biaryl formation (two connected aromatic rings)
            product_mol = Chem.MolFromSmiles(product_smiles)
            has_biaryl = False
            if product_mol:
                biaryl_pattern = "c-c"  # Simple aromatic-aromatic bond
                has_biaryl = product_mol.HasSubstructMatch(Chem.MolFromSmarts(biaryl_pattern))
            
            return has_boronic and has_halide and has_biaryl
            
        except Exception:
            return False
    
    def route_scoring(self, x):
        """
        Score based on timing of Suzuki couplings.
        Better scores for appropriate early-to-mid timing.
        """
        if x < 0:
            return 0  # Condition not met
        
        # x is average depth fraction of Suzuki couplings
        if self.timing == "early_and_mid":
            # Optimal range is 0.2 to 0.6 (early to mid)
            if 0.2 <= x <= 0.6:
                return 1.0
            elif x < 0.2:
                # Too early, but still good
                return 0.8
            elif x <= 0.8:
                # Getting late, but acceptable
                return 0.6 - (x - 0.6) * 2  # Linear decay
            else:
                # Too late
                return 0.2
        
        return 1.0  # Default scoring for other timing constraints
