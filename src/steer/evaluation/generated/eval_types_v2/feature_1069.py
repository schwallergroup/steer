"""Generated evaluation code for: Convergent synthesis via two fragments"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentSynthesis(BaseScoring):
    """
    Evaluates convergent synthesis strategy by detecting when two or more 
    substantial fragments are coupled together at a specific depth.
    
    Checks for amide bond formation from carboxylic acid and amine fragments
    or other coupling reactions that combine pre-formed molecular fragments.
    """
    
    def __init__(self, config: Dict):
        self.fragment_count = config.get("fragment_count", 2)
        self.target_depth = config.get("coupling_depth", 0)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Convergent coupling doesn't happen
        else:
            # Earlier convergent coupling is better (lower depth)
            if x == self.target_depth:
                return 1.0  # Perfect match
            else:
                # Penalize deviation from target depth
                return max(0, 1.0 - 0.2 * abs(x - self.target_depth))
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction represents a convergent coupling step
        by analyzing if multiple substantial fragments are being joined.
        """
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            rxn_parts = rxn_smiles.split(">>")
            
            if len(rxn_parts) != 2:
                return False
                
            product_smiles = rxn_parts[0]
            reactant_smiles = rxn_parts[1]
            
            # Parse reactants
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactant_smiles.split(".")]
            reactants = [mol for mol in reactants if mol is not None]
            
            if len(reactants) < self.fragment_count:
                return False
                
            product = Chem.MolFromSmiles(product_smiles)
            if product is None:
                return False
                
            # Check if this is a coupling reaction by detecting:
            # 1. Amide bond formation (COOH + NH2 -> CONH)
            # 2. Ester formation (COOH + OH -> COO)
            # 3. Other common coupling patterns
            
            substantial_fragments = self._count_substantial_fragments(reactants)
            is_coupling_reaction = self._detect_coupling_reaction(reactants, product)
            
            return substantial_fragments >= self.fragment_count and is_coupling_reaction
            
        except Exception:
            return False
    
    def _count_substantial_fragments(self, reactants) -> int:
        """Count reactants that are substantial molecular fragments (>= 5 heavy atoms)"""
        substantial_count = 0
        for mol in reactants:
            if mol is not None:
                heavy_atom_count = mol.GetNumHeavyAtoms()
                if heavy_atom_count >= 5:  # Threshold for "substantial" fragment
                    substantial_count += 1
        return substantial_count
    
    def _detect_coupling_reaction(self, reactants, product) -> bool:
        """Detect common coupling reaction patterns"""
        try:
            # Check for amide formation (carboxylic acid + amine)
            carboxylic_acid_pattern = Chem.MolFromSmarts("[C](=O)[OH]")
            amine_pattern = Chem.MolFromSmarts("[NH2,NH1]")
            amide_pattern = Chem.MolFromSmarts("[C](=O)[NH]")
            
            has_carboxylic_acid = any(mol.HasSubstructMatch(carboxylic_acid_pattern) 
                                    for mol in reactants if mol is not None)
            has_amine = any(mol.HasSubstructMatch(amine_pattern) 
                          for mol in reactants if mol is not None)
            has_amide = product.HasSubstructMatch(amide_pattern)
            
            if has_carboxylic_acid and has_amine and has_amide:
                return True
            
            # Check for ester formation
            ester_pattern = Chem.MolFromSmarts("[C](=O)[O][C]")
            alcohol_pattern = Chem.MolFromSmarts("[OH]")
            
            has_alcohol = any(mol.HasSubstructMatch(alcohol_pattern) 
                            for mol in reactants if mol is not None)
            has_ester = product.HasSubstructMatch(ester_pattern)
            
            if has_carboxylic_acid and has_alcohol and has_ester:
                return True
            
            # Check for C-C bond forming reactions (simplified)
            # Look for increase in ring count or significant structural changes
            reactant_bonds = sum(mol.GetNumBonds() for mol in reactants if mol is not None)
            product_bonds = product.GetNumBonds()
            
            # If product has significantly fewer bonds than sum of reactants,
            # it's likely a coupling reaction (bonds lost due to condensation)
            if reactant_bonds > product_bonds and (reactant_bonds - product_bonds) >= 1:
                return True
                
            return False
            
        except Exception:
            return False
