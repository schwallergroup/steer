"""Generated evaluation code for: Late stage Boc protection"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageBocProtection(BaseScoring):
    """
    Evaluates whether Boc protection occurs at the final step of synthesis.
    Detects carbamate protection reactions using Boc-Cl or similar reagents.
    """
    
    def __init__(self, config: Dict):
        self.timing = config.get("timing", "final_step")
        # Boc protection reagents
        self.boc_reagents = [
            "CC(C)(C)OC(=O)Cl",  # Boc-Cl
            "CC(C)(C)OC(=O)OC(=O)OC(C)(C)C",  # Boc2O
            "CC(C)(C)OC(=O)ON1C2=CC=CC=C2C2=CC=CC=C21"  # Boc-ONBt
        ]
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Boc protection doesn't occur
        
        if self.timing == "final_step":
            # For final step timing, we want x to be close to 1.0 (very late stage)
            if x >= 0.9:  # Final 10% of synthesis
                return 10
            elif x >= 0.7:  # Late stage
                return 7
            elif x >= 0.5:  # Mid-late stage
                return 4
            else:  # Too early
                return 1
        else:
            # General late-stage preference
            return x * 10
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction node represents a Boc protection reaction.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn:
            return False
        
        try:
            rxn_parts = mapped_rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
            
            reactants_smiles = rxn_parts[0]
            product_smiles = rxn_parts[1]
            
            # Parse reactants and product
            reactant_mols = []
            for smi in reactants_smiles.split("."):
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    reactant_mols.append(mol)
            
            product_mol = Chem.MolFromSmiles(product_smiles)
            if not product_mol or not reactant_mols:
                return False
            
            # Check for Boc reagent in reactants
            has_boc_reagent = False
            for reagent_smiles in self.boc_reagents:
                reagent_mol = Chem.MolFromSmiles(reagent_smiles)
                if reagent_mol:
                    for reactant in reactant_mols:
                        if self._molecules_match(reactant, reagent_mol):
                            has_boc_reagent = True
                            break
                if has_boc_reagent:
                    break
            
            if not has_boc_reagent:
                return False
            
            # Check for carbamate formation pattern
            # Look for Boc group in product: -NHC(=O)OC(C)(C)C
            boc_pattern = Chem.MolFromSmarts("[NH1,NH0][C](=O)[O][C]([CH3])([CH3])[CH3]")
            if not boc_pattern:
                return False
            
            # Check if product contains Boc group and reactant amine doesn't
            product_has_boc = product_mol.HasSubstructMatch(boc_pattern)
            
            # Find the substrate (largest non-reagent reactant)
            substrate = None
            max_atoms = 0
            for reactant in reactant_mols:
                is_reagent = False
                for reagent_smiles in self.boc_reagents:
                    reagent_mol = Chem.MolFromSmiles(reagent_smiles)
                    if reagent_mol and self._molecules_match(reactant, reagent_mol):
                        is_reagent = True
                        break
                
                if not is_reagent and reactant.GetNumAtoms() > max_atoms:
                    substrate = reactant
                    max_atoms = reactant.GetNumAtoms()
            
            if substrate:
                substrate_has_boc = substrate.HasSubstructMatch(boc_pattern)
                # True Boc protection: product has Boc but substrate doesn't
                return product_has_boc and not substrate_has_boc
            
            return product_has_boc
            
        except Exception:
            return False
    
    def _molecules_match(self, mol1, mol2) -> bool:
        """
        Check if two molecules are the same (ignoring stereochemistry and atom mapping).
        """
        try:
            # Remove atom mapping and create canonical SMILES
            mol1_copy = Chem.Mol(mol1)
            mol2_copy = Chem.Mol(mol2)
            
            for atom in mol1_copy.GetAtoms():
                atom.SetAtomMapNum(0)
            for atom in mol2_copy.GetAtoms():
                atom.SetAtomMapNum(0)
            
            smi1 = Chem.MolToSmiles(mol1_copy, canonical=True)
            smi2 = Chem.MolToSmiles(mol2_copy, canonical=True)
            
            return smi1 == smi2
        except Exception:
            return False
