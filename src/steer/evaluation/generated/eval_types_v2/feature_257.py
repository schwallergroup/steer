"""Generated evaluation code for: Late stage heterocycle formation via condensation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageHeterocycleCondensation(BaseScoring):
    """
    Evaluates routes for late-stage heterocycle formation via condensation reactions.
    Detects intramolecular condensation between functional groups like ketone-ester,
    aldehyde-amine, etc. that form heterocyclic rings.
    """
    
    def __init__(self, config: Dict):
        self.step_depth = config.get("step_depth", 2)
        self.condition_type = "depth"
        
        # SMARTS patterns for common condensation-forming heterocycles
        self.heterocycle_patterns = [
            # Pyrazoles, imidazoles
            "[#7]1[#6]=[#6][#7][#6]1",
            "[#7]1[#6]=[#6][#6]=[#7]1",
            # Oxazoles, thiazoles  
            "[#8]1[#6]=[#6][#7][#6]1",
            "[#16]1[#6]=[#6][#7][#6]1",
            # Pyrimidines, pyrazines
            "[#7]1[#6]=[#7][#6]=[#6][#6]1",
            "[#7]1[#6]=[#6][#7]=[#6][#6]1",
            # Lactams, lactones
            "[#7]1[#6](=[#8])[#6][#6][#6]1",
            "[#8]1[#6](=[#8])[#6][#6][#6]1",
            # Benzimidazoles, benzoxazoles
            "[#7]1[#6]=[#7][#6]2[#6]=[#6][#6]=[#6][#6]12",
            "[#8]1[#6]=[#7][#6]2[#6]=[#6][#6]=[#6][#6]12"
        ]
        
        # SMARTS for condensation-prone functional groups
        self.condensation_groups = [
            "[#6](=[#8])[#8][#6]",  # Ester
            "[#6](=[#8])[#6]",      # Ketone
            "[#6]=[#8]",            # Aldehyde
            "[#7][#6]",             # Amine
            "[#8][#6]",             # Alcohol/hydroxyl
            "[#16][#6]"             # Thiol
        ]

    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Condition not met
        
        # Convert depth fraction to score (late stage = higher score)
        if x <= 0.2:  # Very late stage (top 20% of tree)
            return 10
        elif x <= 0.4:  # Late stage  
            return 8
        elif x <= 0.6:  # Mid-late stage
            return 6
        elif x <= 0.8:  # Mid stage
            return 3
        else:  # Early stage
            return 1

    def hit_condition(self, d) -> bool:
        """Check if this reaction represents heterocycle formation via condensation."""
        
        if "mapped_reaction_smiles" not in d.get("metadata", {}):
            return False
            
        rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
        rxn_parts = rxn_smiles.split(">>")
        
        if len(rxn_parts) != 2:
            return False
            
        try:
            product = Chem.MolFromSmiles(rxn_parts[0])
            reactants = [Chem.MolFromSmiles(r) for r in rxn_parts[1].split(".")]
            
            if not product or not all(reactants):
                return False
                
            # Check if product contains heterocycle
            has_heterocycle = any(product.HasSubstructMatch(Chem.MolFromSmarts(pattern)) 
                                for pattern in self.heterocycle_patterns)
            
            if not has_heterocycle:
                return False
                
            # Check if this is likely a condensation reaction
            return self._is_condensation_reaction(product, reactants)
            
        except:
            return False

    def _is_condensation_reaction(self, product, reactants) -> bool:
        """Determine if this is a condensation reaction forming the heterocycle."""
        
        # Count heteroatoms in ring systems
        product_ring_heteroatoms = self._count_ring_heteroatoms(product)
        reactant_ring_heteroatoms = sum(self._count_ring_heteroatoms(r) for r in reactants)
        
        # Heterocycle formation should increase ring heteroatoms
        if product_ring_heteroatoms <= reactant_ring_heteroatoms:
            return False
            
        # Check for presence of condensation-prone functional groups in reactants
        condensation_groups_present = 0
        for reactant in reactants:
            for pattern in self.condensation_groups:
                if reactant.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                    condensation_groups_present += 1
                    break
                    
        # Need at least 2 reactive groups for intramolecular condensation
        # or multiple reactants with complementary groups
        if len(reactants) == 1:
            # Intramolecular - need multiple reactive sites
            return condensation_groups_present >= 2
        else:
            # Intermolecular - each reactant should have complementary groups
            return condensation_groups_present >= len(reactants)

    def _count_ring_heteroatoms(self, mol) -> int:
        """Count heteroatoms that are part of ring systems."""
        ring_info = mol.GetRingInfo()
        ring_atoms = set()
        for ring in ring_info.AtomRings():
            ring_atoms.update(ring)
            
        heteroatom_count = 0
        for atom_idx in ring_atoms:
            atom = mol.GetAtomWithIdx(atom_idx)
            if atom.GetAtomicNum() not in [1, 6]:  # Not H or C
                heteroatom_count += 1
                
        return heteroatom_count
