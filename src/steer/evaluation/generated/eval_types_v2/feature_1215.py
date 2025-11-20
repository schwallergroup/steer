"""Generated evaluation code for: Convergent synthesis via three fragments"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentThreeFragment(MultiRxnCondBase):
    """
    Evaluates convergent synthesis strategy using three fragments assembled 
    via specific coupling reactions (Suzuki and amide couplings).
    """
    
    def __init__(self, config):
        self.fragment_count = config["fragment_count"]
        self.required_reactions = config["assembly_reactions"]
        self.allow_suzuki = "suzuki_coupling" in self.required_reactions
        self.allow_amide = "amide_coupling" in self.required_reactions
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Check for convergent assembly pattern
        convergent_structure = self.detect_convergent_assembly(reactions)
        
        # Check for required coupling reactions
        has_suzuki = any(self.detect_suzuki_coupling(r) for r in reactions)
        has_amide = any(self.detect_amide_coupling(r) for r in reactions)
        
        # Evaluate fragment assembly strategy
        fragment_assembly = self.evaluate_fragment_strategy(reactions)
        
        # Condition is met if we have convergent assembly with required reactions
        condition = (convergent_structure and 
                    has_suzuki == self.allow_suzuki and 
                    has_amide == self.allow_amide and
                    fragment_assembly)
        
        return condition, len(reactions)
    
    def detect_suzuki_coupling(self, rxn):
        """Detect Suzuki coupling reaction pattern"""
        # Suzuki coupling: Ar-B(OR)2 + Ar-X -> Ar-Ar
        boronic_acid = Chem.MolFromSmarts("[c,C]-B(-[OH,O])(-[OH,O])")
        boronic_ester = Chem.MolFromSmarts("[c,C]-B1-O-C-C-O-1")
        aryl_halide = Chem.MolFromSmarts("[c,C]-[Br,I,Cl]")
        
        reactants_smiles = rxn.split(">>")[0]
        reactants = [Chem.MolFromSmiles(r) for r in reactants_smiles.split(".")]
        
        has_boron = False
        has_halide = False
        
        for mol in reactants:
            if mol is None:
                continue
            if (mol.HasSubstructMatch(boronic_acid) or 
                mol.HasSubstructMatch(boronic_ester)):
                has_boron = True
            if mol.HasSubstructMatch(aryl_halide):
                has_halide = True
                
        return has_boron and has_halide
    
    def detect_amide_coupling(self, rxn):
        """Detect amide coupling reaction pattern"""
        # Amide formation: R-COOH + R-NH2 -> R-CO-NH-R
        carboxylic_acid = Chem.MolFromSmarts("[C](=[O])-[OH]")
        amine = Chem.MolFromSmarts("[N;H1,H2]")
        acid_chloride = Chem.MolFromSmarts("[C](=[O])-[Cl]")
        
        reactants_smiles = rxn.split(">>")[0]
        products_smiles = rxn.split(">>")[1]
        reactants = [Chem.MolFromSmiles(r) for r in reactants_smiles.split(".")]
        products = [Chem.MolFromSmiles(p) for p in products_smiles.split(".")]
        
        has_acid_source = False
        has_amine = False
        has_amide_product = False
        
        for mol in reactants:
            if mol is None:
                continue
            if (mol.HasSubstructMatch(carboxylic_acid) or 
                mol.HasSubstructMatch(acid_chloride)):
                has_acid_source = True
            if mol.HasSubstructMatch(amine):
                has_amine = True
        
        # Check for amide formation in products
        amide_pattern = Chem.MolFromSmarts("[C](=[O])-[N]")
        for mol in products:
            if mol is not None and mol.HasSubstructMatch(amide_pattern):
                has_amide_product = True
                
        return has_acid_source and has_amine and has_amide_product
    
    def detect_convergent_assembly(self, reactions):
        """
        Detect if the synthesis follows a convergent strategy by analyzing
        the branching pattern and fragment assembly
        """
        # Look for reactions that combine multiple substantial fragments
        convergent_reactions = 0
        
        for rxn in reactions:
            reactants_smiles = rxn.split(">>")[0]
            reactants = reactants_smiles.split(".")
            
            # Count substantial fragments (>5 heavy atoms)
            substantial_fragments = 0
            for r_smiles in reactants:
                mol = Chem.MolFromSmiles(r_smiles)
                if mol is not None and mol.GetNumHeavyAtoms() > 5:
                    substantial_fragments += 1
            
            # Convergent reaction should combine 2+ substantial fragments
            if substantial_fragments >= 2:
                convergent_reactions += 1
        
        # For 3-fragment strategy, expect at least 2 convergent steps
        return convergent_reactions >= 2
    
    def evaluate_fragment_strategy(self, reactions):
        """
        Evaluate if the synthesis efficiently assembles three key fragments
        """
        # Track the complexity buildup through the synthesis
        total_reactions = len(reactions)
        
        # For efficient 3-fragment assembly, expect moderate number of steps
        # Too few suggests linear, too many suggests inefficient
        if total_reactions < 3:
            return False  # Too simple for 3-fragment strategy
        elif total_reactions > 12:
            return False  # Likely not convergent enough
        
        # Check for balanced fragment sizes in later reactions
        late_stage_reactions = reactions[-min(4, len(reactions)):]
        balanced_assembly = False
        
        for rxn in late_stage_reactions:
            reactants_smiles = rxn.split(">>")[0]
            reactants = reactants_smiles.split(".")
            
            if len(reactants) >= 2:
                sizes = []
                for r_smiles in reactants:
                    mol = Chem.MolFromSmiles(r_smiles)
                    if mol is not None:
                        sizes.append(mol.GetNumHeavyAtoms())
                
                # Check if fragments are reasonably balanced (not too disparate)
                if len(sizes) >= 2:
                    max_size = max(sizes)
                    min_size = min(sizes)
                    if min_size > 3 and max_size / min_size < 4:
                        balanced_assembly = True
                        break
        
        return balanced_assembly
