"""Generated evaluation code for: Late stage aromatic amination"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageAromaticAmination(BaseScoring):
    """
    Evaluates if late-stage aromatic amination occurs via nucleophilic aromatic 
    substitution or Buchwald-Hartwig amination. Returns higher scores when these 
    reactions occur later in the synthesis (closer to final product).
    """
    
    def __init__(self, config: Dict):
        self.timing = config.get("timing", "late")
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Reaction doesn't occur
        else:
            if self.timing == "late":
                return 1 - x  # Later is better (x is depth fraction, so 1-x rewards late stage)
            else:
                return x  # Earlier is better
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction is an aromatic amination (SNAr or Buchwald-Hartwig)"""
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            return self._is_aromatic_amination(rxn_smiles)
        except KeyError:
            return False
    
    def _is_aromatic_amination(self, rxn_smiles: str) -> bool:
        """Detect aromatic amination reactions"""
        try:
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse molecules
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            
            # Filter out None molecules
            products = [p for p in products if p is not None]
            reactants = [r for r in reactants if r is not None]
            
            return self._detect_snar(reactants, products) or self._detect_buchwald_hartwig(reactants, products)
            
        except Exception:
            return False
    
    def _detect_snar(self, reactants, products) -> bool:
        """Detect nucleophilic aromatic substitution forming C-N bond"""
        # Look for aryl halide/nitro + amine -> aryl amine pattern
        aryl_halide_pattern = Chem.MolFromSmarts("[cH0,c:1]-[F,Cl,Br,I,N+]")  # Aromatic C with leaving group
        amine_pattern = Chem.MolFromSmarts("[NH2,NH1,NH0]")  # Nitrogen nucleophile
        aryl_amine_pattern = Chem.MolFromSmarts("c-[NH2,NH1,NH0]")  # Aromatic C-N bond
        
        if not all([aryl_halide_pattern, amine_pattern, aryl_amine_pattern]):
            return False
        
        # Check reactants for aryl halide and amine
        has_aryl_halide = any(mol.HasSubstructMatch(aryl_halide_pattern) for mol in reactants)
        has_amine = any(mol.HasSubstructMatch(amine_pattern) for mol in reactants)
        
        # Check products for aryl amine
        has_aryl_amine = any(mol.HasSubstructMatch(aryl_amine_pattern) for mol in products)
        
        return has_aryl_halide and has_amine and has_aryl_amine
    
    def _detect_buchwald_hartwig(self, reactants, products) -> bool:
        """Detect Buchwald-Hartwig amination (Pd-catalyzed C-N coupling)"""
        # Look for aryl halide + amine -> aryl amine with typical BH conditions
        aryl_halide_pattern = Chem.MolFromSmarts("c-[Br,I,Cl]")  # Aromatic halide
        amine_pattern = Chem.MolFromSmarts("[NH2,NH1]")  # Primary or secondary amine
        aryl_amine_pattern = Chem.MolFromSmarts("c-[NH1,NH0]")  # Aromatic C-N bond
        
        if not all([aryl_halide_pattern, amine_pattern, aryl_amine_pattern]):
            return False
        
        # Check for typical BH pattern: ArX + R2NH -> ArNR2
        has_aryl_halide = any(mol.HasSubstructMatch(aryl_halide_pattern) for mol in reactants)
        has_amine = any(mol.HasSubstructMatch(amine_pattern) for mol in reactants)
        has_aryl_amine = any(mol.HasSubstructMatch(aryl_amine_pattern) for mol in products)
        
        # Additional check: carbon count should be conserved (no C-C bond formation)
        if has_aryl_halide and has_amine and has_aryl_amine:
            reactant_carbons = sum(len([a for a in mol.GetAtoms() if a.GetSymbol() == 'C']) for mol in reactants)
            product_carbons = sum(len([a for a in mol.GetAtoms() if a.GetSymbol() == 'C']) for mol in products)
            return abs(reactant_carbons - product_carbons) <= 1  # Allow for small variations
        
        return False
