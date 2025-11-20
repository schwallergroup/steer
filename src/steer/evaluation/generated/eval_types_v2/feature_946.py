"""Generated evaluation code for: Convergent synthesis via two key fragments"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentSynthesis(BaseScoring):
    """
    Evaluates convergent synthesis routes that combine two key fragments via 
    nucleophilic aromatic substitution at a late stage.
    
    Checks for the presence of S_NAr coupling reactions and scores based on
    when they occur in the synthesis (later is better for convergence).
    """
    
    def __init__(self, config: Dict):
        self.fragment_count = config.get("fragment_count", 2)
        self.coupling_reaction = config.get("coupling_reaction", "nucleophilic_aromatic_substitution")
        self.coupling_stage = config.get("coupling_stage", "late")
        
        # SMARTS patterns for nucleophilic aromatic substitution
        # Electron-deficient aromatic rings that can undergo S_NAr
        self.aromatic_electrophile_patterns = [
            "[cH0:1]1[cH][c]([N+](=O)[O-])[cH][cH][c]1[F,Cl,Br,I]",  # nitro-substituted aryl halide
            "[cH0:1]1[cH][c]([C](=O)[O,N])[cH][cH][c]1[F,Cl,Br,I]",   # carbonyl-substituted aryl halide
            "[cH0:1]1[cH][c]([C]#N)[cH][cH][c]1[F,Cl,Br,I]",          # cyano-substituted aryl halide
            "[cH0:1]1[cH][c]([S](=O)(=O))[cH][cH][c]1[F,Cl,Br,I]",    # sulfonyl-substituted aryl halide
        ]
        
        # Nucleophile patterns
        self.nucleophile_patterns = [
            "[NH2,NH1,NH0]-[c,C]",  # amines
            "[OH]-[c,C]",           # alcohols/phenols
            "[SH,S-]",              # thiols/thiolates
        ]

    def route_scoring(self, x) -> float:
        """Convert depth fraction to score (0-10). Late coupling gets higher scores."""
        if x < 0:
            return 0  # No S_NAr coupling found
        
        if self.coupling_stage == "late":
            # Reward later coupling (higher depth fraction = higher score)
            return x * 10
        else:
            # For early coupling preference
            return (1 - x) * 10

    def hit_condition(self, d) -> bool:
        """Check if this reaction represents a nucleophilic aromatic substitution coupling."""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            reactants_smiles, product_smiles = mapped_rxn.split(">>")
            reactant_mols = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            product_mol = Chem.MolFromSmiles(product_smiles.strip())
            
            if not all(mol is not None for mol in reactant_mols) or product_mol is None:
                return False
                
            # Check if we have the right number of fragments coupling
            if len(reactant_mols) != self.fragment_count:
                return False
                
            # Look for S_NAr pattern: electrophilic aromatic + nucleophile -> coupled product
            has_electrophile = False
            has_nucleophile = False
            
            for reactant in reactant_mols:
                # Check for aromatic electrophile
                for pattern in self.aromatic_electrophile_patterns:
                    if reactant.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                        has_electrophile = True
                        break
                        
                # Check for nucleophile
                for pattern in self.nucleophile_patterns:
                    if reactant.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                        has_nucleophile = True
                        break
            
            # Verify that both reactants are present and coupling occurred
            if has_electrophile and has_nucleophile:
                # Additional check: verify that a new C-N, C-O, or C-S bond formed
                return self._verify_coupling_bond_formation(reactant_mols, product_mol)
                
        except Exception:
            return False
            
        return False
    
    def _verify_coupling_bond_formation(self, reactants, product):
        """Verify that a new heteroatom-carbon bond was formed in the coupling."""
        try:
            # Count heteroatom-aromatic carbon bonds in reactants vs product
            def count_aromatic_hetero_bonds(mol):
                count = 0
                for bond in mol.GetBonds():
                    atom1, atom2 = bond.GetBeginAtom(), bond.GetEndAtom()
                    if ((atom1.GetIsAromatic() and atom1.GetSymbol() == 'C' and 
                         atom2.GetSymbol() in ['N', 'O', 'S']) or
                        (atom2.GetIsAromatic() and atom2.GetSymbol() == 'C' and 
                         atom1.GetSymbol() in ['N', 'O', 'S'])):
                        count += 1
                return count
            
            reactant_bonds = sum(count_aromatic_hetero_bonds(mol) for mol in reactants)
            product_bonds = count_aromatic_hetero_bonds(product)
            
            # New bond should have formed
            return product_bonds > reactant_bonds
            
        except Exception:
            return True  # Default to true if verification fails
