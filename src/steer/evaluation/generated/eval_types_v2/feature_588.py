"""Generated evaluation code for: Missing protecting group for piperidine nitrogen"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class MissingPiperidineProtection(MultiRxnCondBase):
    """
    Evaluates synthesis routes for missing protecting groups on piperidine nitrogen
    during critical reactions like amide coupling and nitrile dehydration.
    Returns penalty score based on presence of unprotected piperidine during these reactions.
    """
    
    def __init__(self, config):
        self.target_reactions = config.get("reaction_steps", ["amide_coupling", "nitrile_dehydration"])
        self.penalty_weight = config.get("penalty_weight", 1.0)
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        total_reactions = len(reactions)
        
        violations = 0
        for rxn in reactions:
            if self.is_critical_reaction(rxn) and self.has_unprotected_piperidine(rxn):
                violations += 1
        
        # Condition is met (good) if no violations found
        condition_met = violations == 0
        violation_fraction = violations / max(1, total_reactions)
        
        return condition_met, violation_fraction
    
    def route_scoring(self, x):
        if isinstance(x, tuple):
            condition_met, violation_fraction = x
            if condition_met:
                return 0  # No penalty
            else:
                return violation_fraction * 10 * self.penalty_weight
        else:
            # Fallback for simple scoring
            return x * 10 * self.penalty_weight
    
    def is_critical_reaction(self, rxn):
        """Check if reaction is amide coupling or nitrile dehydration"""
        try:
            rxn_parts = rxn.split(">>")
            reactants = rxn_parts[0]
            products = rxn_parts[1]
            
            # Detect amide coupling: carboxylic acid/ester + amine -> amide
            if self.detect_amide_coupling(reactants, products):
                return True
                
            # Detect nitrile dehydration: amide -> nitrile
            if self.detect_nitrile_dehydration(reactants, products):
                return True
                
            return False
        except:
            return False
    
    def detect_amide_coupling(self, reactants, products):
        """Detect amide bond formation"""
        try:
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".")]
            
            # Check for carboxylic acid or activated ester in reactants
            carboxyl_pattern = Chem.MolFromSmarts("[CX3](=O)[OX2H1,OX1-]")  # Carboxylic acid
            ester_pattern = Chem.MolFromSmarts("[CX3](=O)[OX2][CX4]")  # Ester
            amide_pattern = Chem.MolFromSmarts("[CX3](=O)[NX3]")  # Amide
            
            has_carboxyl_reactant = any(mol and mol.HasSubstructMatch(carboxyl_pattern) for mol in reactant_mols if mol)
            has_ester_reactant = any(mol and mol.HasSubstructMatch(ester_pattern) for mol in reactant_mols if mol)
            has_amide_product = any(mol and mol.HasSubstructMatch(amide_pattern) for mol in product_mols if mol)
            
            return (has_carboxyl_reactant or has_ester_reactant) and has_amide_product
        except:
            return False
    
    def detect_nitrile_dehydration(self, reactants, products):
        """Detect nitrile formation from amide dehydration"""
        try:
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".")]
            
            amide_pattern = Chem.MolFromSmarts("[CX3](=O)[NX3]")  # Amide
            nitrile_pattern = Chem.MolFromSmarts("[CX2]#[NX1]")  # Nitrile
            
            has_amide_reactant = any(mol and mol.HasSubstructMatch(amide_pattern) for mol in reactant_mols if mol)
            has_nitrile_product = any(mol and mol.HasSubstructMatch(nitrile_pattern) for mol in product_mols if mol)
            
            return has_amide_reactant and has_nitrile_product
        except:
            return False
    
    def has_unprotected_piperidine(self, rxn):
        """Check if reaction involves unprotected piperidine nitrogen"""
        try:
            rxn_parts = rxn.split(">>")
            all_molecules = rxn_parts[0] + "." + rxn_parts[1]
            
            for mol_smiles in all_molecules.split("."):
                mol = Chem.MolFromSmiles(mol_smiles.strip())
                if mol and self.contains_unprotected_piperidine(mol):
                    return True
            return False
        except:
            return False
    
    def contains_unprotected_piperidine(self, mol):
        """Check if molecule contains unprotected piperidine nitrogen"""
        try:
            # Piperidine ring pattern
            piperidine_pattern = Chem.MolFromSmarts("[NX3;R1][CX4;R1][CX4;R1][CX4;R1][CX4;R1][CX4;R1]")
            
            if not mol.HasSubstructMatch(piperidine_pattern):
                return False
            
            # Check if piperidine nitrogen is protected
            # Protected nitrogen patterns: amide, carbamate, sulfonamide
            protected_n_patterns = [
                Chem.MolFromSmarts("[NX3;R1][CX3](=O)"),  # Amide protection
                Chem.MolFromSmarts("[NX3;R1][CX3](=O)[OX2]"),  # Carbamate protection (Boc, Cbz, etc.)
                Chem.MolFromSmarts("[NX3;R1][SX4](=O)(=O)"),  # Sulfonamide protection
            ]
            
            matches = mol.GetSubstructMatches(piperidine_pattern)
            for match in matches:
                n_idx = match[0]  # Nitrogen index
                n_atom = mol.GetAtomWithIdx(n_idx)
                
                # Check if this nitrogen is protected
                is_protected = False
                for pattern in protected_n_patterns:
                    if mol.HasSubstructMatch(pattern):
                        protected_matches = mol.GetSubstructMatches(pattern)
                        for p_match in protected_matches:
                            if n_idx in p_match:
                                is_protected = True
                                break
                    if is_protected:
                        break
                
                if not is_protected:
                    return True
            
            return False
        except:
            return False
