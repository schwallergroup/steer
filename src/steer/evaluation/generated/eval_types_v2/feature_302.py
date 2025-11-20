"""Generated evaluation code for: Convergent synthesis via two Suzuki couplings"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentSuzukiStrategy(MultiRxnCondBase):
    """
    Evaluates convergent synthesis strategy using two Suzuki coupling reactions.
    Checks if the route contains exactly two Suzuki couplings that assemble 
    major fragments in a convergent manner.
    """
    
    def __init__(self, config):
        self.required_suzuki_count = config.get("coupling_count", 2)
        self.require_fragment_assembly = config.get("fragment_assembly", True)
        
        # Suzuki coupling detection patterns
        self.boronic_acid_pattern = Chem.MolFromSmarts("[#6]-B(O)O")
        self.boronic_ester_pattern = Chem.MolFromSmarts("[#6]-B1OC(C)(C)C(C)(C)O1")
        self.aryl_halide_pattern = Chem.MolFromSmarts("[#6][Cl,Br,I]")
        self.biaryl_pattern = Chem.MolFromSmarts("c-c")
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        suzuki_reactions = []
        for i, rxn in enumerate(reactions):
            if self.detect_suzuki_coupling(rxn):
                suzuki_reactions.append((i, rxn))
        
        # Check if we have exactly the required number of Suzuki couplings
        has_correct_count = len(suzuki_reactions) == self.required_suzuki_count
        
        # Check convergent assembly if required
        convergent_assembly = True
        if self.require_fragment_assembly and len(suzuki_reactions) >= 2:
            convergent_assembly = self.check_convergent_assembly(suzuki_reactions, reactions)
        
        condition_met = has_correct_count and convergent_assembly
        total_reactions = len(reactions)
        
        return condition_met, total_reactions
    
    def detect_suzuki_coupling(self, rxn):
        """Detect if a reaction is a Suzuki coupling based on reactants and products."""
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants_smiles = rxn_parts[0]
            products_smiles = rxn_parts[1]
            
            # Parse reactants
            reactant_mols = []
            for r_smi in reactants_smiles.split("."):
                mol = Chem.MolFromSmiles(r_smi)
                if mol:
                    reactant_mols.append(mol)
            
            # Parse products
            product_mols = []
            for p_smi in products_smiles.split("."):
                mol = Chem.MolFromSmiles(p_smi)
                if mol:
                    product_mols.append(mol)
            
            if len(reactant_mols) < 2 or len(product_mols) < 1:
                return False
            
            # Check for boronic acid/ester and aryl halide in reactants
            has_boron_component = any(
                mol.HasSubstructMatch(self.boronic_acid_pattern) or 
                mol.HasSubstructMatch(self.boronic_ester_pattern)
                for mol in reactant_mols
            )
            
            has_halide_component = any(
                mol.HasSubstructMatch(self.aryl_halide_pattern)
                for mol in reactant_mols
            )
            
            # Check for biaryl formation in products
            has_biaryl_product = any(
                mol.HasSubstructMatch(self.biaryl_pattern)
                for mol in product_mols
            )
            
            return has_boron_component and has_halide_component and has_biaryl_product
            
        except Exception:
            return False
    
    def check_convergent_assembly(self, suzuki_reactions, all_reactions):
        """
        Check if Suzuki couplings represent convergent assembly of major fragments.
        This checks that the couplings occur at different stages and involve 
        substantial molecular fragments.
        """
        if len(suzuki_reactions) < 2:
            return False
        
        # Check that Suzuki reactions are not consecutive (indicating convergence)
        suzuki_indices = [idx for idx, _ in suzuki_reactions]
        
        # Ensure reactions are spread out in the synthesis
        if len(suzuki_indices) >= 2:
            # Check minimum separation between Suzuki reactions
            min_separation = min(abs(suzuki_indices[i] - suzuki_indices[j]) 
                               for i in range(len(suzuki_indices)) 
                               for j in range(i+1, len(suzuki_indices)))
            
            # Require at least one reaction step between Suzuki couplings
            if min_separation < 2:
                return False
        
        # Check molecular complexity of fragments being coupled
        for _, rxn in suzuki_reactions:
            if not self.involves_substantial_fragments(rxn):
                return False
        
        return True
    
    def involves_substantial_fragments(self, rxn):
        """Check if the Suzuki coupling involves substantial molecular fragments."""
        try:
            rxn_parts = rxn.split(">>")
            reactants_smiles = rxn_parts[0]
            
            reactant_mols = []
            for r_smi in reactants_smiles.split("."):
                mol = Chem.MolFromSmiles(r_smi)
                if mol:
                    reactant_mols.append(mol)
            
            # Check that at least two reactants have substantial size (>6 heavy atoms)
            substantial_fragments = sum(1 for mol in reactant_mols 
                                      if mol.GetNumHeavyAtoms() > 6)
            
            return substantial_fragments >= 2
            
        except Exception:
            return False
