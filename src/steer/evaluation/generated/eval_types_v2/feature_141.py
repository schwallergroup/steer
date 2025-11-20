"""Generated evaluation code for: Late stage C-N coupling assembly"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageCNCoupling(BaseScoring):
    """
    Evaluates synthesis routes for late-stage C-N coupling reactions.
    Specifically looks for aryl-amine bond formation via palladium-catalyzed
    C-N coupling reactions occurring late in the synthesis.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "float")
        self.target_depth = config.get("target_depth", {}).get("value", 0.1)  # Default to very late stage
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # C-N coupling doesn't happen
        else:
            # Late-stage coupling is better, score inversely with depth
            # x is depth fraction (0 = final step, 1 = earliest step)
            if self.condition_type == "bool":
                # Return 1 if coupling occurs in final 20% of route
                return 1 if x <= 0.2 else 0
            else:
                # Continuous scoring - penalize early C-N coupling
                return max(0, 1 - (x / self.target_depth))
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction step represents a C-N coupling forming an aryl-amine bond.
        """
        metadata = d.get("metadata", {})
        
        # Check for C-N coupling policy/template if available
        policy_name = metadata.get("policy_name", "")
        if "c-n" in policy_name.lower() or "buchwald" in policy_name.lower() or "ullmann" in policy_name.lower():
            return True
            
        # Check reaction SMILES for C-N bond formation pattern
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            prod_smiles, react_smiles = mapped_rxn.split(">>")
            prod_mol = Chem.MolFromSmiles(prod_smiles)
            react_mols = [Chem.MolFromSmiles(r.strip()) for r in react_smiles.split(".")]
            
            if not prod_mol or not all(react_mols):
                return False
                
            # Look for aryl-amine pattern in product
            aryl_amine_patterns = [
                "[cH0:1][NH1:2][CH]",  # aromatic C-N-aliphatic
                "[cH0:1][NH1:2][c]",   # aromatic C-N-aromatic  
                "[cH0:1][NH0:2]([CH])[CH]",  # aromatic C-N(alkyl)2
                "[cH0:1][NH0:2]([c])[CH]",   # aromatic C-N(aryl)(alkyl)
            ]
            
            for pattern in aryl_amine_patterns:
                patt_mol = Chem.MolFromSmarts(pattern)
                if patt_mol and prod_mol.HasSubstructMatch(patt_mol):
                    # Check if this C-N bond is newly formed (not present in reactants)
                    matches = prod_mol.GetSubstructMatches(patt_mol)
                    for match in matches:
                        c_map = prod_mol.GetAtomWithIdx(match[0]).GetAtomMapNum()
                        n_map = prod_mol.GetAtomWithIdx(match[1]).GetAtomMapNum()
                        
                        if c_map > 0 and n_map > 0:
                            # Check if C and N atoms are in different reactant molecules
                            c_in_react = []
                            n_in_react = []
                            
                            for i, react_mol in enumerate(react_mols):
                                c_found = any(a.GetAtomMapNum() == c_map for a in react_mol.GetAtoms())
                                n_found = any(a.GetAtomMapNum() == n_map for a in react_mol.GetAtoms())
                                if c_found:
                                    c_in_react.append(i)
                                if n_found:
                                    n_in_react.append(i)
                            
                            # C-N coupling if C and N come from different reactants
                            if c_in_react and n_in_react and not any(c == n for c in c_in_react for n in n_in_react):
                                return True
                                
            return False
            
        except Exception:
            return False
