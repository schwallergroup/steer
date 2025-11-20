"""Generated evaluation code for: Convergent amide coupling strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentAmideCoupling(BaseScoring):
    """
    Evaluates convergent amide coupling strategy where two complex fragments 
    are joined via amide bond formation using acid chloride and amine coupling.
    """
    
    def __init__(self, config: Dict):
        self.bond_type = config["parameters"]["bond_type"]
        self.fragments = config["parameters"]["fragments"]
        self.coupling_reaction = config["parameters"]["coupling_reaction"]
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Amide coupling doesn't happen
        else:
            # Earlier convergent coupling is better (more strategic)
            return 1 - x
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction represents convergent amide coupling
        between acid chloride and amine fragments.
        """
        if "mapped_reaction_smiles" not in d.get("metadata", {}):
            return False
            
        rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
        rxn_parts = rxn_smiles.split(">>")
        
        if len(rxn_parts) != 2:
            return False
            
        products = rxn_parts[0]
        reactants = rxn_parts[1].split(".")
        
        # Need exactly 2 main reactant fragments for convergent strategy
        if len(reactants) < 2:
            return False
            
        try:
            prod_mol = Chem.MolFromSmiles(products)
            reactant_mols = [Chem.MolFromSmiles(r) for r in reactants if r]
            
            if not prod_mol or len(reactant_mols) < 2:
                return False
            
            # Check for amide bond formation
            if not self._has_amide_formation(prod_mol, reactant_mols):
                return False
            
            # Check for acid chloride and amine coupling pattern
            if not self._is_acid_chloride_amine_coupling(reactant_mols):
                return False
            
            # Verify fragments are reasonably complex (convergent strategy)
            if not self._are_fragments_complex(reactant_mols):
                return False
                
            return True
            
        except Exception:
            return False
    
    def _has_amide_formation(self, product, reactants) -> bool:
        """Check if an amide bond is formed in the reaction."""
        # Amide pattern: C(=O)N
        amide_pattern = Chem.MolFromSmarts("[C](=O)[N]")
        if not amide_pattern:
            return False
            
        # Product should have amide bond
        if not product.HasSubstructMatch(amide_pattern):
            return False
            
        # Count amide bonds in product vs reactants
        prod_amides = len(product.GetSubstructMatches(amide_pattern))
        reactant_amides = sum(len(r.GetSubstructMatches(amide_pattern)) for r in reactants)
        
        # Should have net formation of at least one amide bond
        return prod_amides > reactant_amides
    
    def _is_acid_chloride_amine_coupling(self, reactants) -> bool:
        """Check for acid chloride + amine coupling pattern."""
        # Acid chloride pattern: C(=O)Cl
        acid_chloride_pattern = Chem.MolFromSmarts("[C](=O)[Cl]")
        # Primary or secondary amine pattern: [N;H1,H2]
        amine_pattern = Chem.MolFromSmarts("[N;H1,H2;!$(N=*);!$(N#*)]")
        
        if not acid_chloride_pattern or not amine_pattern:
            return False
        
        has_acid_chloride = False
        has_amine = False
        
        for reactant in reactants:
            if reactant.HasSubstructMatch(acid_chloride_pattern):
                has_acid_chloride = True
            if reactant.HasSubstructMatch(amine_pattern):
                has_amine = True
                
        return has_acid_chloride and has_amine
    
    def _are_fragments_complex(self, reactants) -> bool:
        """
        Check if the coupling fragments are reasonably complex
        to qualify as convergent strategy.
        """
        main_fragments = []
        
        for reactant in reactants:
            # Skip small molecules (solvents, reagents)
            if reactant.GetNumHeavyAtoms() >= 8:  # Arbitrary threshold for "complex"
                main_fragments.append(reactant)
        
        # Need at least 2 complex fragments for convergent coupling
        return len(main_fragments) >= 2
