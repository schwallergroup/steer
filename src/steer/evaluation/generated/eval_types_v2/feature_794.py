"""Generated evaluation code for: Convergent synthesis via two fragments"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentSynthesis(MultiRxnCondBase):
    """
    Evaluates convergent synthesis strategy where two fragments are built separately
    and then coupled via a specific reaction type at a particular stage.
    """
    
    def __init__(self, config):
        self.fragment_count = config.get("fragment_count", 2)
        self.coupling_reaction = config.get("coupling_reaction", "SNAr")
        self.coupling_stage = config.get("coupling_stage", "late")  # "early", "middle", "late"
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        total_reactions = len(reactions)
        
        if total_reactions == 0:
            return False, 0
        
        # Find the coupling reaction
        coupling_depth = -1
        for i, rxn in enumerate(reactions):
            if self.detect_coupling_reaction(rxn):
                coupling_depth = i
                break
        
        if coupling_depth == -1:
            return False, total_reactions
        
        # Check if coupling occurs at the desired stage
        stage_condition = self.check_coupling_stage(coupling_depth, total_reactions)
        
        # Check if the coupling reaction joins the right number of fragments
        fragment_condition = self.check_fragment_convergence(reactions[coupling_depth])
        
        condition = stage_condition and fragment_condition
        return condition, coupling_depth + 1
    
    def detect_coupling_reaction(self, rxn):
        """Detect specific coupling reaction type based on structural patterns"""
        if self.coupling_reaction.lower() == "snar":
            return self.detect_snar_reaction(rxn)
        elif self.coupling_reaction.lower() == "suzuki":
            return self.detect_suzuki_coupling(rxn)
        elif self.coupling_reaction.lower() == "buchwald":
            return self.detect_buchwald_hartwig(rxn)
        elif self.coupling_reaction.lower() == "amide":
            return self.detect_amide_coupling(rxn)
        else:
            return False
    
    def detect_snar_reaction(self, rxn):
        """Detect nucleophilic aromatic substitution"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
        
        product = Chem.MolFromSmiles(rxn_parts[0])
        reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[1].split(".")]
        
        if not product or len(reactants) < 2:
            return False
        
        # Look for aromatic halide + nucleophile pattern
        aromatic_halide_pattern = Chem.MolFromSmarts("[cH0,c:1][F,Cl,Br,I]")
        nucleophile_pattern = Chem.MolFromSmarts("[NH2,NH1,OH,SH]")
        
        has_ar_halide = any(mol.HasSubstructMatch(aromatic_halide_pattern) for mol in reactants)
        has_nucleophile = any(mol.HasSubstructMatch(nucleophile_pattern) for mol in reactants)
        
        return has_ar_halide and has_nucleophile
    
    def detect_suzuki_coupling(self, rxn):
        """Detect Suzuki-Miyaura coupling"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
        
        reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[1].split(".")]
        if len(reactants) < 2:
            return False
        
        # Look for aryl halide + boronic acid/ester pattern
        aryl_halide_pattern = Chem.MolFromSmarts("[c][Br,I,Cl]")
        boronic_pattern = Chem.MolFromSmarts("[c][B]([OH])([OH])")
        
        has_halide = any(mol and mol.HasSubstructMatch(aryl_halide_pattern) for mol in reactants if mol)
        has_boronic = any(mol and mol.HasSubstructMatch(boronic_pattern) for mol in reactants if mol)
        
        return has_halide and has_boronic
    
    def detect_buchwald_hartwig(self, rxn):
        """Detect Buchwald-Hartwig amination"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
        
        reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[1].split(".")]
        if len(reactants) < 2:
            return False
        
        # Look for aryl halide + amine pattern
        aryl_halide_pattern = Chem.MolFromSmarts("[c][Br,I,Cl]")
        amine_pattern = Chem.MolFromSmarts("[NH2,NH1]")
        
        has_halide = any(mol and mol.HasSubstructMatch(aryl_halide_pattern) for mol in reactants if mol)
        has_amine = any(mol and mol.HasSubstructMatch(amine_pattern) for mol in reactants if mol)
        
        return has_halide and has_amine
    
    def detect_amide_coupling(self, rxn):
        """Detect amide bond formation"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
        
        product = Chem.MolFromSmiles(rxn_parts[0])
        reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[1].split(".")]
        
        if not product or len(reactants) < 2:
            return False
        
        # Look for carboxylic acid/ester + amine pattern
        carboxyl_pattern = Chem.MolFromSmarts("[C](=[O])[OH,O]")
        amine_pattern = Chem.MolFromSmarts("[NH2,NH1]")
        amide_pattern = Chem.MolFromSmarts("[C](=[O])[NH]")
        
        has_carboxyl = any(mol and mol.HasSubstructMatch(carboxyl_pattern) for mol in reactants if mol)
        has_amine = any(mol and mol.HasSubstructMatch(amine_pattern) for mol in reactants if mol)
        has_amide_product = product.HasSubstructMatch(amide_pattern)
        
        return has_carboxyl and has_amine and has_amide_product
    
    def check_coupling_stage(self, coupling_depth, total_reactions):
        """Check if coupling occurs at the desired stage"""
        if total_reactions <= 1:
            return True
        
        relative_position = coupling_depth / (total_reactions - 1)
        
        if self.coupling_stage == "early":
            return relative_position <= 0.33
        elif self.coupling_stage == "middle":
            return 0.33 < relative_position <= 0.67
        elif self.coupling_stage == "late":
            return relative_position > 0.67
        else:
            return True
    
    def check_fragment_convergence(self, rxn):
        """Check if the reaction joins the expected number of fragments"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
        
        # Count non-reagent reactants (exclude small molecules like catalysts, bases)
        reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[1].split(".")]
        significant_reactants = [
