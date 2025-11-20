"""Generated evaluation code for: Convergent synthesis via two fragments"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentSynthesis(BaseScoring):
    """
    Evaluates convergent synthesis strategy where two complex fragments are constructed 
    separately and joined via a coupling reaction (e.g., Suzuki-Miyaura coupling).
    
    Checks for the presence of the specified coupling reaction and ensures it occurs
    between two substantial fragments rather than simple starting materials.
    """
    
    def __init__(self, config: Dict):
        self.fragment_count = config["fragment_count"]
        self.coupling_reaction = config["coupling_reaction"].lower()
        
        # Define coupling reaction patterns
        self.coupling_patterns = {
            "suzuki": {
                "boronic_acid": "[#6][B]([OH])[OH]",  # Boronic acid
                "boronate": "[#6][B]1OC(C)(C)C(C)(C)O1",  # Pinacol boronate
                "halide": "[#6][Cl,Br,I]",  # Aryl/vinyl halide
                "triflate": "[#6]OS(=O)(=O)C(F)(F)F"  # Triflate
            },
            "buchwald": {
                "amine": "[NX3;H2,H1;!$(NC=O)]",  # Primary/secondary amine
                "halide": "[#6][Cl,Br,I]",
                "triflate": "[#6]OS(=O)(=O)C(F)(F)F"
            },
            "heck": {
                "alkene": "[#6]=[#6]",  # Alkene
                "halide": "[#6][Cl,Br,I]",
                "triflate": "[#6]OS(=O)(=O)C(F)(F)F"
            }
        }
        
        # Minimum heavy atom count for a "substantial" fragment
        self.min_fragment_size = 8
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Coupling reaction not found
        else:
            # Earlier convergent coupling is generally better
            return max(0, 10 * (1 - x))
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction represents the desired coupling reaction."""
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            rxn_parts = rxn_smiles.split(">>")
            
            if len(rxn_parts) != 2:
                return False
                
            products = rxn_parts[0]
            reactants = rxn_parts[1].split(".")
            
            # Need at least 2 reactants for coupling
            if len(reactants) < 2:
                return False
            
            # Check if this is the specified coupling reaction
            if not self._is_coupling_reaction(reactants, products):
                return False
            
            # Verify we have substantial fragments (not just simple starting materials)
            substantial_fragments = self._count_substantial_fragments(reactants)
            
            return substantial_fragments >= self.fragment_count
            
        except Exception:
            return False
    
    def _is_coupling_reaction(self, reactants: List[str], products: str) -> bool:
        """Check if the reaction matches the specified coupling type."""
        if self.coupling_reaction not in self.coupling_patterns:
            return False
        
        patterns = self.coupling_patterns[self.coupling_reaction]
        reactant_mols = [Chem.MolFromSmiles(r) for r in reactants if Chem.MolFromSmiles(r)]
        
        if self.coupling_reaction == "suzuki":
            return self._check_suzuki_coupling(reactant_mols)
        elif self.coupling_reaction == "buchwald":
            return self._check_buchwald_coupling(reactant_mols)
        elif self.coupling_reaction == "heck":
            return self._check_heck_coupling(reactant_mols)
        
        return False
    
    def _check_suzuki_coupling(self, reactant_mols: List) -> bool:
        """Check for Suzuki coupling: organoborane + organohalide."""
        has_borane = False
        has_halide = False
        
        for mol in reactant_mols:
            if mol is None:
                continue
                
            # Check for boronic acid or boronate
            if (mol.HasSubstructMatch(Chem.MolFromSmarts(self.coupling_patterns["suzuki"]["boronic_acid"])) or
                mol.HasSubstructMatch(Chem.MolFromSmarts(self.coupling_patterns["suzuki"]["boronate"]))):
                has_borane = True
            
            # Check for halide or triflate
            if (mol.HasSubstructMatch(Chem.MolFromSmarts(self.coupling_patterns["suzuki"]["halide"])) or
                mol.HasSubstructMatch(Chem.MolFromSmarts(self.coupling_patterns["suzuki"]["triflate"]))):
                has_halide = True
        
        return has_borane and has_halide
    
    def _check_buchwald_coupling(self, reactant_mols: List) -> bool:
        """Check for Buchwald-Hartwig coupling: amine + organohalide."""
        has_amine = False
        has_halide = False
        
        for mol in reactant_mols:
            if mol is None:
                continue
                
            if mol.HasSubstructMatch(Chem.MolFromSmarts(self.coupling_patterns["buchwald"]["amine"])):
                has_amine = True
            
            if (mol.HasSubstructMatch(Chem.MolFromSmarts(self.coupling_patterns["buchwald"]["halide"])) or
                mol.HasSubstructMatch(Chem.MolFromSmarts(self.coupling_patterns["buchwald"]["triflate"]))):
                has_halide = True
        
        return has_amine and has_halide
    
    def _check_heck_coupling(self, reactant_mols: List) -> bool:
        """Check for Heck coupling: alkene + organohalide."""
        has_alkene = False
        has_halide = False
        
        for mol in reactant_mols:
            if mol is None:
                continue
                
            if mol.HasSubstructMatch(Chem.MolFromSmarts(self.coupling_patterns["heck"]["alkene"])):
                has_alkene = True
            
            if (mol.HasSubstructMatch(Chem.MolFromSmarts(self.coupling_patterns["heck"]["halide"])) or
                mol.HasSubstructMatch(Chem.MolFromSmarts(self.coupling_patterns["heck"]["triflate"]))):
                has_halide = True
        
        return has_alkene and has_halide
    
    def _count_substantial_fragments(self, reactants: List[str]) -> int:
        """Count reactants that are substantial fragments (not simple starting materials)."""
        substantial_count = 0
        
        for reactant_smiles in reactants:
            mol = Chem.MolFromSmiles(reactant_smiles)
            if mol is None:
                continue
                
            heavy_atom_count = mol.GetNumHeavyAtoms()
            if heavy_atom_count >= self.min_fragment_size:
                substantial_count += 1
        
        return substantial_count
