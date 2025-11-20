"""Generated evaluation code for: Late stage Suzuki coupling for aryl installation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageSuzukiCoupling(BaseScoring):
    """
    Evaluates whether a Suzuki-Miyaura coupling reaction occurs in the late stages
    of a synthesis route (within the specified stage threshold from the target).
    
    Returns higher scores when Suzuki coupling is used closer to the final target,
    with 0 if no Suzuki coupling is detected.
    """
    
    def __init__(self, config: Dict):
        self.stage_threshold = config["parameters"].get("stage_threshold", 0.2)
    
    def route_scoring(self, x) -> float:
        """
        Convert depth fraction to score (0-10).
        Higher scores for reactions closer to target (lower depth fraction).
        """
        if x < 0:
            return 0  # No Suzuki coupling found
        
        if x <= self.stage_threshold:
            # Late stage coupling - highest score for latest stages
            return 10 * (1 - x / self.stage_threshold)
        else:
            # Early stage coupling - lower score
            return max(0, 3 * (1 - x))
    
    def hit_condition(self, d) -> bool:
        """
        Check if the reaction is a Suzuki-Miyaura coupling.
        Detects formation of C-C bonds between aryl/heteroaryl groups.
        """
        metadata = d.get("metadata", {})
        
        # Check for common Suzuki coupling patterns in reaction SMILES
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        if not mapped_rxn:
            return False
        
        try:
            rxn_parts = mapped_rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
            
            products = rxn_parts[0]
            reactants = rxn_parts[1].split(".")
            
            # Look for boronic acid/ester patterns in reactants
            boronic_patterns = [
                "[#6]B(O)O",  # Boronic acid
                "[#6]B1OC(C)(C)C(C)(C)O1",  # Pinacol boronate
                "[#6]B(O[#6])[O#6]",  # Boronic ester
                "[#6][B]([OH])[OH]",  # Alternative boronic acid
            ]
            
            # Look for aryl halide patterns
            halide_patterns = [
                "[#6]~[#6]-[Br,I,Cl]",  # Aryl halide
                "c-[Br,I,Cl]",  # Aromatic halide
            ]
            
            has_boronic = False
            has_halide = False
            
            for reactant_smiles in reactants:
                try:
                    reactant_mol = Chem.MolFromSmiles(reactant_smiles)
                    if reactant_mol is None:
                        continue
                    
                    # Check for boronic acid/ester
                    for pattern in boronic_patterns:
                        patt_mol = Chem.MolFromSmarts(pattern)
                        if patt_mol and reactant_mol.HasSubstructMatch(patt_mol):
                            has_boronic = True
                            break
                    
                    # Check for aryl halide
                    for pattern in halide_patterns:
                        patt_mol = Chem.MolFromSmarts(pattern)
                        if patt_mol and reactant_mol.HasSubstructMatch(patt_mol):
                            has_halide = True
                            break
                    
                    if has_boronic and has_halide:
                        break
                        
                except:
                    continue
            
            # Additional check: look for new C-C bond formation between aromatic systems
            if has_boronic and has_halide:
                return True
            
            # Alternative detection: check for typical Suzuki reaction conditions/reagents
            reactant_string = " ".join(reactants).lower()
            suzuki_indicators = ["b(oh)", "bpin", "boronic", "pd", "palladium"]
            
            if any(indicator in reactant_string for indicator in suzuki_indicators):
                # Verify C-C bond formation by checking aromatic carbon count increase
                try:
                    prod_mol = Chem.MolFromSmiles(products)
                    total_reactant_atoms = sum(len(Chem.MolFromSmiles(r).GetAtoms()) 
                                             for r in reactants if Chem.MolFromSmiles(r))
                    
                    if prod_mol and len(prod_mol.GetAtoms()) < total_reactant_atoms:
                        return True
                except:
                    pass
            
            return False
            
        except Exception:
            return False
