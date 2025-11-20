"""Generated evaluation code for: Late stage phenyl installation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStagePhenyInstallation(BaseScoring):
    """
    Evaluates synthesis routes for late-stage phenyl installation via Suzuki coupling.
    Rewards routes where a phenyl group is added late in the synthesis through
    Suzuki coupling with an aryl halide substrate.
    """
    
    def __init__(self, config: Dict):
        self.timing_preference = config.get("timing", "late")  # "late" or "early"
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Suzuki coupling doesn't happen
        else:
            if self.timing_preference == "late":
                return 1 - x  # Later is better, x is depth fraction (0=early, 1=late)
            else:
                return x  # Earlier is better
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction node represents a Suzuki coupling installing a phenyl group
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            product_smiles = rxn_parts[0]
            reactants_smiles = rxn_parts[1]
            
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r) for r in reactants_smiles.split(".") if r]
            
            if not product or not all(reactants):
                return False
            
            # Check for Suzuki coupling pattern: aryl halide + phenylboronic acid/ester
            has_aryl_halide = False
            has_phenyl_boron = False
            
            # Patterns for aryl halides (Br, I, Cl on aromatic carbon)
            aryl_halide_patterns = [
                Chem.MolFromSmarts("[cH0,cH1:1]-[Br]"),
                Chem.MolFromSmarts("[cH0,cH1:1]-[I]"),
                Chem.MolFromSmarts("[cH0,cH1:1]-[Cl]")
            ]
            
            # Patterns for phenylboronic acid/esters
            phenyl_boron_patterns = [
                Chem.MolFromSmarts("c1ccccc1-[B](-[OH])-[OH]"),  # phenylboronic acid
                Chem.MolFromSmarts("c1ccccc1-[B]1-[O]-[C]-[C]-[O]-1"),  # phenylboronic pinacol ester
                Chem.MolFromSmarts("c1ccccc1-[B]"),  # general phenyl-boron
            ]
            
            # Check reactants for required patterns
            for reactant in reactants:
                # Check for aryl halide substrate
                for pattern in aryl_halide_patterns:
                    if reactant.HasSubstructMatch(pattern):
                        has_aryl_halide = True
                        break
                
                # Check for phenylboronic acid/ester
                for pattern in phenyl_boron_patterns:
                    if reactant.HasSubstructMatch(pattern):
                        has_phenyl_boron = True
                        break
            
            # Verify that product contains a new phenyl-aryl bond
            if has_aryl_halide and has_phenyl_boron:
                # Additional check: product should have more phenyl rings than aryl halide reactant
                phenyl_pattern = Chem.MolFromSmarts("c1ccccc1")
                product_phenyl_count = len(product.GetSubstructMatches(phenyl_pattern))
                
                aryl_halide_reactant = None
                for reactant in reactants:
                    for pattern in aryl_halide_patterns:
                        if reactant.HasSubstructMatch(pattern):
                            aryl_halide_reactant = reactant
                            break
                    if aryl_halide_reactant:
                        break
                
                if aryl_halide_reactant:
                    reactant_phenyl_count = len(aryl_halide_reactant.GetSubstructMatches(phenyl_pattern))
                    return product_phenyl_count > reactant_phenyl_count
            
            return False
            
        except Exception:
            return False
