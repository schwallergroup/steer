"""Generated evaluation code for: Convergent synthesis via Suzuki coupling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentSuzukiCoupling(BaseScoring):
    """
    Evaluates convergent synthesis routes that use Suzuki coupling as a key step.
    Checks for the presence of Suzuki coupling reaction and evaluates its position
    in the synthesis route for optimal convergency.
    """
    
    def __init__(self, config: Dict):
        self.fragment_count = config["parameters"].get("fragment_count", 2)
        self.coupling_step_position = config["parameters"].get("coupling_step_position", "middle")
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # No Suzuki coupling found
        
        # Score based on position preference
        if self.coupling_step_position == "middle":
            # Prefer coupling in the middle of the route (around 0.3-0.7)
            if 0.3 <= x <= 0.7:
                return 10  # Optimal position
            elif 0.1 <= x <= 0.9:
                return 7   # Good position
            else:
                return 4   # Suboptimal but present
        elif self.coupling_step_position == "early":
            # Prefer early coupling (x < 0.5)
            if x <= 0.3:
                return 10
            elif x <= 0.5:
                return 7
            else:
                return 4
        elif self.coupling_step_position == "late":
            # Prefer late coupling (x > 0.5)
            if x >= 0.7:
                return 10
            elif x >= 0.5:
                return 7
            else:
                return 4
        else:
            # Default: any position is good if Suzuki is present
            return 8
    
    def hit_condition(self, d):
        """Check if this reaction node represents a Suzuki coupling."""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn:
            return False
            
        # Split reaction SMILES
        if ">>" not in mapped_rxn:
            return False
            
        reactants_str, product_str = mapped_rxn.split(">>")
        reactants = reactants_str.split(".")
        
        # Check if we have at least the expected number of fragments
        if len(reactants) < self.fragment_count:
            return False
            
        try:
            product_mol = Chem.MolFromSmiles(product_str)
            reactant_mols = [Chem.MolFromSmiles(r) for r in reactants if r.strip()]
            
            if not product_mol or not all(reactant_mols):
                return False
                
            # Check for Suzuki coupling characteristics
            return self._is_suzuki_coupling(reactant_mols, product_mol)
            
        except Exception:
            return False
    
    def _is_suzuki_coupling(self, reactants, product):
        """Detect Suzuki coupling by looking for characteristic patterns."""
        
        # Boronic acid/ester patterns
        boronic_patterns = [
            "[#6][B]([OH])[OH]",  # Boronic acid
            "[#6][B]1OCC(C)(C)CO1",  # Pinacol boronate
            "[#6][B](F)(F)F",  # Trifluoroborate
            "[#6][B]([O-])[O-]",  # Boronate anion
        ]
        
        # Halide patterns (for aryl/vinyl halides)
        halide_patterns = [
            "[#6][Br]",  # Aryl/alkyl bromide
            "[#6][I]",   # Aryl/alkyl iodide
            "[#6][Cl]",  # Aryl/alkyl chloride
        ]
        
        has_boron_component = False
        has_halide_component = False
        
        # Check reactants for boronic compounds and halides
        for reactant in reactants:
            # Check for boronic acid/ester
            for pattern in boronic_patterns:
                if reactant.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                    has_boron_component = True
                    break
                    
            # Check for halides
            for pattern in halide_patterns:
                if reactant.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                    has_halide_component = True
                    break
        
        # Basic requirement: need both boron and halide components
        if not (has_boron_component and has_halide_component):
            return False
            
        # Additional check: look for C-C bond formation
        # Count carbon atoms in reactants vs product
        reactant_carbons = sum(mol.GetNumAtoms() for mol in reactants 
                              for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)
        product_carbons = sum(1 for atom in product.GetAtoms() if atom.GetAtomicNum() == 6)
        
        # In Suzuki coupling, we typically form new C-C bonds
        # The carbon count should be preserved (minus any leaving groups)
        return abs(reactant_carbons - product_carbons) <= 2  # Allow for small differences due to leaving groups
