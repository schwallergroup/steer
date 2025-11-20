"""Generated evaluation code for: Convergent synthesis via three fragments"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentSynthesis(MultiRxnCondBase):
    """
    Evaluates convergent synthesis routes that assemble multiple fragments 
    through specific coupling reactions like amide or ether formation.
    """
    
    def __init__(self, config):
        self.fragment_count = config.get("fragment_count", 3)
        self.coupling_points = config.get("coupling_points", ["amide_formation", "ether_formation"])
        
        # Define SMARTS patterns for coupling reactions
        self.coupling_patterns = {
            "amide_formation": "[C:1](=[O:2])[NH:3]",
            "ether_formation": "[C,c:1][O:2][C,c:3]",
            "ester_formation": "[C:1](=[O:2])[O:3][C,c:4]",
            "carbon_carbon": "[C:1][C:2]",
            "suzuki_coupling": "[c:1][c:2]"  # aromatic C-C
        }
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        """
        Checks if the route represents convergent synthesis with the specified
        fragment count and coupling reactions.
        """
        reactions = self.get_rxns(d)
        
        # Check if we have coupling reactions of the specified types
        coupling_reactions = []
        for rxn in reactions:
            if self.is_coupling_reaction(rxn):
                coupling_reactions.append(rxn)
        
        # Analyze convergence - count distinct fragments being assembled
        fragment_analysis = self.analyze_convergence(reactions)
        
        # Condition is met if:
        # 1. We have enough coupling reactions
        # 2. The synthesis is convergent (multiple fragments assembled)
        # 3. Fragment count matches target
        condition = (
            len(coupling_reactions) >= 2 and
            fragment_analysis["is_convergent"] and
            fragment_analysis["max_fragments"] >= self.fragment_count
        )
        
        return condition, len(reactions)
    
    def is_coupling_reaction(self, rxn):
        """
        Determines if a reaction involves formation of bonds typical in 
        convergent coupling reactions.
        """
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants = rxn_parts[0].split(".")
            products = rxn_parts[1].split(".")
            
            # Skip if not combining multiple reactants into fewer products
            if len(reactants) < 2 or len(products) > len(reactants):
                return False
            
            # Check for specific coupling patterns in products
            for product_smiles in products:
                try:
                    product_mol = Chem.MolFromSmiles(product_smiles)
                    if product_mol is None:
                        continue
                        
                    for coupling_type in self.coupling_points:
                        if coupling_type in self.coupling_patterns:
                            pattern = Chem.MolFromSmarts(self.coupling_patterns[coupling_type])
                            if pattern and product_mol.HasSubstructMatch(pattern):
                                return True
                except:
                    continue
                    
            return False
            
        except:
            return False
    
    def analyze_convergence(self, reactions):
        """
        Analyzes the reaction sequence to determine convergence characteristics.
        """
        result = {
            "is_convergent": False,
            "max_fragments": 1,
            "coupling_steps": 0
        }
        
        # Track molecule count through the synthesis
        molecule_counts = []
        
        for rxn in reactions:
            try:
                rxn_parts = rxn.split(">>")
                if len(rxn_parts) != 2:
                    continue
                    
                reactants = rxn_parts[0].split(".")
                products = rxn_parts[1].split(".")
                
                # Count non-trivial reactants (exclude small molecules like water, etc.)
                significant_reactants = [r for r in reactants if self.is_significant_molecule(r)]
                
                molecule_counts.append(len(significant_reactants))
                
                # If combining 2+ significant molecules, it's a coupling step
                if len(significant_reactants) >= 2 and len(products) < len(significant_reactants):
                    result["coupling_steps"] += 1
                    
            except:
                continue
        
        if molecule_counts:
            result["max_fragments"] = max(molecule_counts)
            # Convergent if we ever combine multiple fragments
            result["is_convergent"] = result["max_fragments"] >= 2 and result["coupling_steps"] >= 1
            
        return result
    
    def is_significant_molecule(self, smiles):
        """
        Determines if a molecule is significant (not a small reagent/solvent).
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False
                
            # Consider molecules with >= 6 heavy atoms as significant fragments
            heavy_atom_count = mol.GetNumHeavyAtoms()
            return heavy_atom_count >= 6
            
        except:
            return False
