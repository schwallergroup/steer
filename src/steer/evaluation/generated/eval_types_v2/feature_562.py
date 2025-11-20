"""Generated evaluation code for: Convergent synthesis via late stage coupling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentStrategy(BaseScoring):
    """
    Evaluates convergent synthesis strategy by detecting late-stage coupling reactions
    that join two complex fragments. Rewards routes where complex fragments are
    coupled at depths greater than the specified threshold.
    """
    
    def __init__(self, config: Dict):
        self.coupling_stage_threshold = config["coupling_stage_threshold"]
        self.fragment_complexity_min = config["fragment_complexity_min"]
        
        # Common coupling reaction SMARTS patterns
        self.coupling_patterns = [
            # Suzuki coupling - Ar-B + Ar-X -> Ar-Ar
            "[#6]~[#5].[#6]~[#6]>>",
            # Sonogashira coupling - alkyne + aryl halide
            "[#6]#[#6].[#6]~[#17,#35,#53]>>",
            # Negishi coupling - organozinc + organic halide  
            "[#6]~[#30].[#6]~[#17,#35,#53]>>",
            # Stille coupling - organotin + organic halide
            "[#6]~[#50].[#6]~[#17,#35,#53]>>",
            # Heck coupling - alkene + aryl halide
            "[#6]=[#6].[#6]~[#17,#35,#53]>>",
            # Ullmann coupling - two aryl halides
            "[#6]~[#17,#35,#53].[#6]~[#17,#35,#53]>>"
        ]

    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # No convergent coupling found
        else:
            # Reward late-stage coupling (higher depth fraction is better)
            if x >= self.coupling_stage_threshold:
                return 10 * (1 - (1 - x) / (1 - self.coupling_stage_threshold))
            else:
                return 5 * (x / self.coupling_stage_threshold)

    def hit_condition(self, d) -> bool:
        """Check if this reaction is a convergent coupling of complex fragments"""
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
                
            reactants_smiles, product_smiles = rxn_smiles.split(">>")
            reactants = reactants_smiles.split(".")
            
            # Must have exactly 2 reactants for convergent coupling
            if len(reactants) != 2:
                return False
                
            # Check if this matches a coupling reaction pattern
            is_coupling = self._is_coupling_reaction(rxn_smiles)
            if not is_coupling:
                return False
                
            # Check fragment complexity
            reactant_mols = [Chem.MolFromSmiles(r) for r in reactants]
            if None in reactant_mols:
                return False
                
            complexities = [self._calculate_complexity(mol) for mol in reactant_mols]
            
            # Both fragments must meet minimum complexity threshold
            return all(c >= self.fragment_complexity_min for c in complexities)
            
        except Exception:
            return False

    def _is_coupling_reaction(self, rxn_smiles: str) -> bool:
        """Check if reaction matches known coupling patterns"""
        try:
            # Simple heuristic: look for patterns indicating C-C bond formation
            reactants, product = rxn_smiles.split(">>")
            
            # Check for organometallic reagents or halides in reactants
            has_organometallic = any(metal in reactants for metal in ['B', 'Zn', 'Sn'])
            has_halide = any(hal in reactants for hal in ['Br', 'Cl', 'I'])
            
            # For Suzuki-like: one reactant has boron, other has halide
            reactant_list = reactants.split(".")
            if len(reactant_list) == 2:
                r1_has_metal = any(metal in reactant_list[0] for metal in ['B', 'Zn', 'Sn'])
                r2_has_metal = any(metal in reactant_list[1] for metal in ['B', 'Zn', 'Sn'])
                r1_has_hal = any(hal in reactant_list[0] for hal in ['Br', 'Cl', 'I'])
                r2_has_hal = any(hal in reactant_list[1] for hal in ['Br', 'Cl', 'I'])
                
                return (r1_has_metal and r2_has_hal) or (r2_has_metal and r1_has_hal)
            
            return has_organometallic and has_halide
            
        except Exception:
            return False

    def _calculate_complexity(self, mol) -> float:
        """Calculate fragment complexity based on heavy atoms, rings, and heteroatoms"""
        if mol is None:
            return 0
            
        heavy_atoms = mol.GetNumHeavyAtoms()
        ring_info = mol.GetRingInfo()
        num_rings = ring_info.NumRings()
        
        # Count heteroatoms (non-C, non-H)
        heteroatoms = sum(1 for atom in mol.GetAtoms() 
                         if atom.GetAtomicNum() not in [1, 6])
        
        # Complexity score: weighted combination
        complexity = heavy_atoms + (2 * num_rings) + (1.5 * heteroatoms)
        return complexity
