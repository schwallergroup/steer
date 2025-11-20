"""Generated evaluation code for: Late stage Suzuki-Miyaura cross-coupling for phenyl installation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageSuzukiPhenyl(BaseScoring):
    """
    Evaluates synthesis routes for late-stage Suzuki-Miyaura cross-coupling 
    reactions that install phenyl groups. Returns higher scores when Suzuki 
    coupling occurs later in the synthesis (closer to final product).
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "fraction")
        self.target_depth = config.get("target_depth", {}).get("value", 0.8)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # No Suzuki coupling found
        else:
            # Late-stage (higher x values) get better scores
            # x is depth fraction, so values closer to 1.0 are later
            return x * 10  # Scale to 0-10 range
    
    def hit_condition(self, d):
        """
        Checks if a reaction node represents a Suzuki-Miyaura coupling
        that installs a phenyl group.
        """
        metadata = d.get("metadata", {})
        rxn_smiles = metadata.get("mapped_reaction_smiles", "")
        
        if not rxn_smiles or ">>" not in rxn_smiles:
            return False
            
        try:
            # Parse reaction SMILES
            rxn_parts = rxn_smiles.split(">>")
            product_smiles = rxn_parts[0]
            reactant_smiles = rxn_parts[1]
            
            product_mol = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r) for r in reactant_smiles.split(".")]
            
            if not product_mol or not all(reactants):
                return False
            
            # Check for Suzuki coupling pattern: boronic acid/ester + aryl halide
            has_boron_reactant = False
            has_halide_reactant = False
            has_phenyl_installation = False
            
            # Define patterns for Suzuki reactants
            boronic_acid_pattern = Chem.MolFromSmarts("[cH1:1][B]([OH])[OH]")  # Phenylboronic acid
            boronic_ester_pattern = Chem.MolFromSmarts("[cH1:1][B]1OC(C)(C)C(C)(C)O1")  # Phenyl boronic ester
            aryl_halide_pattern = Chem.MolFromSmarts("[c,C][F,Cl,Br,I]")
            phenyl_pattern = Chem.MolFromSmarts("c1ccccc1")
            
            # Check reactants for Suzuki coupling components
            for reactant in reactants:
                if (boronic_acid_pattern and reactant.HasSubstructMatch(boronic_acid_pattern)) or \
                   (boronic_ester_pattern and reactant.HasSubstructMatch(boronic_ester_pattern)):
                    has_boron_reactant = True
                elif aryl_halide_pattern and reactant.HasSubstructMatch(aryl_halide_pattern):
                    has_halide_reactant = True
            
            # Check if phenyl group is installed in product
            if phenyl_pattern and product_mol.HasSubstructMatch(phenyl_pattern):
                # Verify phenyl wasn't already present in non-boron reactants
                phenyl_in_non_boron = False
                for reactant in reactants:
                    if reactant.HasSubstructMatch(aryl_halide_pattern) and \
                       reactant.HasSubstructMatch(phenyl_pattern):
                        # Check if phenyl is on the halide-bearing carbon
                        matches = reactant.GetSubstructMatches(aryl_halide_pattern)
                        if matches:
                            phenyl_in_non_boron = True
                            break
                
                if not phenyl_in_non_boron:
                    has_phenyl_installation = True
            
            # Also check common Suzuki coupling reaction names/templates
            policy_name = metadata.get("policy_name", "").lower()
            template_name = metadata.get("template", "").lower()
            
            is_suzuki_by_name = any(keyword in policy_name or keyword in template_name 
                                  for keyword in ["suzuki", "cross_coupling", "coupling"])
            
            return (has_boron_reactant and has_halide_reactant and has_phenyl_installation) or \
                   (is_suzuki_by_name and has_phenyl_installation)
                   
        except Exception:
            return False
