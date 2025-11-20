"""Generated evaluation code for: Late stage biphenyl fragment coupling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageBiphenylCoupling(BaseScoring):
    """
    Evaluates synthesis routes for late-stage biphenyl fragment coupling.
    Rewards routes where a pre-formed biphenyl fragment is coupled with an alkyl chain
    in the final steps via cross-coupling reactions.
    """
    
    def __init__(self, config: Dict):
        self.coupling_position = config.get("coupling_step_position", "final")
        self.fragment_types = config.get("fragment_types", ["aryl_halide", "alkyl_halide"])
        self.coupling_reaction = config.get("coupling_reaction", "cross_coupling")
        
        # SMARTS patterns for detection
        self.biphenyl_pattern = "c1ccc(-c2ccccc2)cc1"  # Basic biphenyl
        self.aryl_halide_patterns = [
            "c-[Cl,Br,I]",  # Aryl halides
            "c-B([OH])([OH])",  # Boronic acids
            "c-B1OC(C)(C)C(C)(C)O1"  # Pinacol boronate esters
        ]
        self.alkyl_halide_patterns = [
            "[CH2,CH]-[Cl,Br,I]",  # Alkyl halides
            "C-[CH2]-[Zn,Mg]",  # Organometallic reagents
        ]

    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Coupling doesn't happen
        else:
            # Reward later coupling (smaller depth fraction is better)
            if x <= 0.2:  # Very late stage (within first 20% of route)
                return 10
            elif x <= 0.4:  # Late stage
                return 8
            elif x <= 0.6:  # Mid-late stage
                return 5
            else:  # Early stage coupling
                return 2

    def hit_condition(self, d) -> bool:
        """Check if this reaction represents biphenyl fragment coupling"""
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
                
            product_smiles, reactants_smiles = rxn_smiles.split(">>")
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r) for r in reactants_smiles.split(".") if r.strip()]
            
            if not product or not reactants or len(reactants) < 2:
                return False
            
            # Check if product contains biphenyl
            biphenyl_in_product = product.HasSubstructMatch(Chem.MolFromSmarts(self.biphenyl_pattern))
            if not biphenyl_in_product:
                return False
            
            # Check if we have appropriate coupling partners in reactants
            has_aryl_component = False
            has_alkyl_component = False
            
            for reactant in reactants:
                # Check for biphenyl-containing aryl halide/boronic acid
                if reactant.HasSubstructMatch(Chem.MolFromSmarts(self.biphenyl_pattern)):
                    for pattern in self.aryl_halide_patterns:
                        if reactant.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                            has_aryl_component = True
                            break
                
                # Check for alkyl halide/organometallic
                for pattern in self.alkyl_halide_patterns:
                    if reactant.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                        has_alkyl_component = True
                        break
            
            # Check if this looks like a cross-coupling reaction
            is_cross_coupling = self._is_cross_coupling_reaction(d)
            
            return has_aryl_component and has_alkyl_component and is_cross_coupling
            
        except Exception:
            return False

    def _is_cross_coupling_reaction(self, d) -> bool:
        """Check if reaction metadata suggests cross-coupling"""
        metadata = d.get("metadata", {})
        
        # Check policy name for cross-coupling indicators
        policy_name = metadata.get("policy_name", "").lower()
        cross_coupling_keywords = [
            "suzuki", "heck", "sonogashira", "stille", "negishi", 
            "kumada", "cross_coupling", "coupling"
        ]
        
        if any(keyword in policy_name for keyword in cross_coupling_keywords):
            return True
        
        # Check reaction template or other metadata
        template = metadata.get("template", "").lower()
        if any(keyword in template for keyword in cross_coupling_keywords):
            return True
            
        return False
