import reAdd commentMore actions
import copy
from collections import Counter, defaultdict
from itertools import product

from colorama import Fore
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, rdFMCS

from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

class MoleculeSetTooCrazyError(Exception):
    """Base class for other custom exceptions"""
    pass

class MoleculeSet:
    """
    Class for a set of one or more molecules
    """

    # Class attribute to know if the default canonicalization is Explicit or RDKit
    default_canonicalization = "Explicit" #"RDKit"

    dict_bond_type = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
        4: Chem.BondType.QUADRUPLE,
    }

    fp_gen = AllChem.GetRDKitFPGenerator()

    def __init__(
        self,
        smiles,
        parent_mol=None,
    ):
        self.smiles = smiles
        # We remove aromaticity in order to avoid problems with the dict_bond_type
        self.mol = Chem.MolFromSmiles(self.smiles)
        self.parent_mol = parent_mol
        try:
            Chem.Kekulize(self.mol, clearAromaticFlags=True)
            self.structure_too_crazy = False
            self.mol = Chem.AddHs(self.mol)
            # Clean way to detect indistinguishable atoms, there can be holes in the numbering,
            # but similar idx means indistinguishable and different idx means distinguishable,
            # Useful to avoid counting twice moves that are indistinguishable.
            self.indistinguishable_idx = Chem.CanonicalRankAtoms(
                self.mol, breakTies=False
            )
            self.n_atoms = self.mol.GetNumAtoms()
        except Exception as e:
            self.structure_too_crazy = True
            raise MoleculeSetTooCrazyError(
                f"Structure too crazy: {self.smiles}, couldn't kekulize: {e}"
            )

        self._fp = None
        self._all_legal_moves = None
        self._all_legal_next_ms = None
        self._all_legal_next_tracked_ms = None
        self._all_legal_next_smiles = None
        self._all_legal_next_tracked_smiles = None
        self._all_legal_next_reactive_center_smiles = None

    def __repr__(self):
        return f'Steer.MoleculeSet("{Chem.MolToSmiles(self.mol)}", smiles="{self.smiles}")'

    @property
    def fp(self):
        if self._fp is None:
            self._fp = MoleculeSet.fp_gen.GetFingerprint(self.mol)
        return self._fp

    @property
    def canonical_smiles(self):
        if MoleculeSet.default_canonicalization == "Explicit":
            return self.explicit_canonical_smiles
        elif MoleculeSet.default_canonicalization == "RDKit":
            return self.rdkit_canonical_smiles
        else:
            raise ValueError(
                f"Default canonicalization method {MoleculeSet.default_canonicalization} not recognized. Please set it to 'Explicit' or 'RDKit'."
            )

    @property
    def rdkit_canonical_smiles(self):
        return Chem.MolToSmiles(Chem.MolFromSmiles(self.explicit_canonical_smiles))

    @property
    def explicit_canonical_smiles(self):
        return self.clean_smiles_artefacts(Chem.MolToSmiles(self.mol))

    @property
    def tracked_smiles(self):
        # copy mol
        mol = copy.deepcopy(self.mol)

        # align Atom Map Idx on tracking_idx
        for a in mol.GetAtoms():
            a.SetAtomMapNum(int(a.GetProp("tracking_idx")))

        return Chem.MolToSmiles(mol)

    @property
    def reactive_center_smiles(self):
        # copy mol
        mol = copy.deepcopy(self.mol)

        if not mol.GetAtomWithIdx(0).HasProp("reactive_center"):
            print(f"{Fore.YELLOW}Error in reactive_center_smiles: MoleculeSet was probably not created with track_atoms=True.{Fore.RESET}")
            return ""

        # align Atom Map Idx on reactive_center_idx
        for a in mol.GetAtoms():
            a.SetAtomMapNum(int(a.GetProp("reactive_center")))
        
        # Same for parent_mol
        parent_mol = copy.deepcopy(self.parent_mol)
        if not parent_mol.GetAtomWithIdx(0).HasProp("reactive_center"):
            print(f"{Fore.YELLOW}Error in reactive_center_smiles: MoleculeSet was probably not created with track_atoms=True.{Fore.RESET}")
            return ""
        # align Atom Map Idx on reactive_center_idx
        for a in parent_mol.GetAtoms():
            a.SetAtomMapNum(int(a.GetProp("reactive_center")))

        return f"{Chem.MolToSmiles(parent_mol)}>>{Chem.MolToSmiles(mol)}"


    @property
    def all_legal_moves(self):
        if self._all_legal_moves is None:
            self._all_legal_moves = self.calculate_all_legal_moves()
        return self._all_legal_moves

    @property
    def all_legal_next_ms(self):
        if self._all_legal_next_ms is None:
            self._all_legal_next_ms = [
                self.make_move(m) for m in self.all_legal_moves
            ]
        return self._all_legal_next_ms

    @property
    def all_legal_next_smiles(self):
        if self._all_legal_next_smiles is None:
            self._all_legal_next_smiles = [
                m.canonical_smiles for m in self.all_legal_next_ms
            ]
        return self._all_legal_next_smiles

    @property
    def all_legal_next_tracked_ms(self):
        if self._all_legal_next_tracked_ms is None:
            self._all_legal_next_tracked_ms = [
                self.make_move(m, track_atoms=True) for m in self.all_legal_moves
            ]
        return self._all_legal_next_tracked_ms

    @property
    def all_legal_next_tracked_smiles(self):
        if self._all_legal_next_tracked_smiles is None:
            self._all_legal_next_tracked_smiles = [
                m.tracked_smiles for m in self.all_legal_next_tracked_ms
            ]
        return self._all_legal_next_tracked_smiles

    @property
    def all_legal_next_reactive_center_smiles(self):
        if self._all_legal_next_tracked_smiles is None:
            self._all_legal_next_tracked_smiles = [
                m.reactive_center_smiles for m in self.all_legal_next_tracked_ms
            ]
        return self._all_legal_next_tracked_smiles

    def filter_indistinguishable_moves(self, list_moves):
        already_observed_move = []
        list_indistinguishable_moves = []

        for move in list_moves:
            # For ionizing as well as for attacking, the two last indices are the two atoms involved in the move, hence these indices -1 and -2
            if (
                not (
                    indistinguishable_move := (
                        self.indistinguishable_idx[move[-2]],
                        self.indistinguishable_idx[move[-1]],
                    )
                )
                in already_observed_move
            ):
                already_observed_move.append(indistinguishable_move)
                list_indistinguishable_moves.append(move)

        return list_indistinguishable_moves

    def determine_lone_pairs(self):
        lone_pairs_dict = {}
        for atom in self.mol.GetAtoms():
            # Total valence electrons for a neutral atom
            atomic_number = atom.GetAtomicNum()
            valence_electrons = Chem.GetPeriodicTable().GetNOuterElecs(
                atomic_number
            )

            # Adjust for atom's formal charge
            valence_electrons -= atom.GetFormalCharge()

            # Calculate number of electrons not involved in bonding
            bonding_electrons = 0
            bonding_electrons = sum(
                [bond.GetBondTypeAsDouble() for bond in atom.GetBonds()]
            )

            non_bonding_electrons = valence_electrons - bonding_electrons

            # Lone pairs are half of the non-bonding electrons
            lone_pairs = non_bonding_electrons // 2
            lone_pairs_dict[atom.GetIdx()] = (atom.GetSymbol(), lone_pairs)
        return [
            (i, lone_pairs_dict[i])
            for i in lone_pairs_dict.keys()
            if lone_pairs_dict[i][1] > 0
        ]

    def determine_empty_orbitals(self):
        lone_pairs_dict = {}
        empty_orbital_dict = {}

        for atom in self.mol.GetAtoms():
            # Total valence electrons for a neutral atom
            atomic_number = atom.GetAtomicNum()
            valence_electrons = Chem.GetPeriodicTable().GetNOuterElecs(
                atomic_number
            )

            # Adjust for atom's formal charge
            valence_electrons -= atom.GetFormalCharge()

            # Calculate number of electrons not involved in bonding
            bonding_electrons = 0
            bonding_electrons = sum(
                [bond.GetBondTypeAsDouble() for bond in atom.GetBonds()]
            )

            non_bonding_electrons = valence_electrons - bonding_electrons

            # lone pairs are half of the non-bonding electrons
            lone_pairs = non_bonding_electrons // 2
            lone_pairs_dict[atom.GetIdx()] = (atom.GetSymbol(), lone_pairs)
            # print(f'This is the number of lone pairs {lone_pairs}')

            # empty orbitals are 2(for first period) or 8(for second period) or 12(for the third period) - ((#bonding_electrons * 2) + lone_pair)
            # divide by 2 because each orbital is 2 electrons
            empty_orbitals = 0
            if atom.GetAtomicNum() <= 2:
                empty_orbitals = (
                    2 - ((bonding_electrons * 2) + lone_pairs * 2)
                ) / 2
            # handle only P and S
            elif atom.GetAtomicNum() == 15 or atom.GetAtomicNum() == 16:
                empty_orbitals = (
                    12 - ((bonding_electrons * 2) + lone_pairs * 2)
                ) / 2
            # treat everything else with octet rule (mainly for halogens)
            else:
                empty_orbitals = (
                    8 - ((bonding_electrons * 2) + lone_pairs * 2)
                ) / 2

            empty_orbital_dict[atom.GetIdx()] = (
                atom.GetSymbol(),
                empty_orbitals,
            )

        return [
            (i, empty_orbital_dict[i])
            for i in empty_orbital_dict.keys()
            if empty_orbital_dict[i][1] > 0
        ]

    def calculate_all_legal_moves(self):
        moves = []
        moves += [("a", m[0], m[1]) for m in self.all_legal_attacking_moves()]
        moves += [("i", m[0], m[1]) for m in self.all_legal_ionizing_moves()]

        return moves

    def all_legal_ionizing_moves(self):
        """
        Returns all moves that would reduce the degree of a bond of 1 to charge positively and negatively its two ends
        A move is described by the index of the bond, the index of the atom that would be charged negatively.
        """

        moves = []

        for bond in self.mol.GetBonds():
            if bond.GetBondTypeAsDouble() >= 1:
                atom1 = bond.GetBeginAtom()
                atom2 = bond.GetEndAtom()
                # The index of the atom that would be charged positively is
                # added to facilitate "filter_indistinguishable_moves" down
                # the line, but it will be removed in "calculate_all_legal_moves"
                moves.append((bond.GetIdx(), atom1.GetIdx(), atom2.GetIdx()))
                moves.append((bond.GetIdx(), atom2.GetIdx(), atom1.GetIdx()))

        moves = self.filter_indistinguishable_moves(moves)
        return moves

    def all_legal_attacking_moves(self):
        """
        Returns all moves of electrons from a lone pair to an empty orbital
        A move is described by the index of the atom with the lone pair and the index of the atom with the empty orbital
        """
        moves = []
        lone_pairs = self.determine_lone_pairs()
        empty_orbitals = self.determine_empty_orbitals()

        for lp, eo in product(lone_pairs, empty_orbitals):
            if lp[0] != eo[0]:  # Avoid attacking moves between the same atom
                moves.append((lp[0], eo[0]))

        moves = self.filter_indistinguishable_moves(moves)
        return moves

    def make_move(self, move, track_atoms=False):

        if move[0] == "i":  # ionization move
            mol = self.make_ionizing_move(*move[1:3], track_atoms=track_atoms)

        elif move[0] == "a":  # attack move
            mol = self.make_attacking_move(*move[1:3], track_atoms=track_atoms)

        else:
            raise ValueError(f"Move type '{move[0]}' not recognized")

        raw_smiles = Chem.MolToSmiles(mol.GetMol())
        cleaned_smiles = self.clean_smiles_artefacts(raw_smiles)
        new_molecule_set = MoleculeSet(
            cleaned_smiles,
        )

        if track_atoms:
            new_molecule_set.align_oneself_with(mol.GetMol())
            new_molecule_set.tag_reactive_center(self.mol)
            new_molecule_set.parent_mol = copy.deepcopy(self.mol)
        return new_molecule_set

    def tag_reactive_center(self, mol):
        # Tag the reactive center as the atoms that changed their direct environment
        
        dict_mol_env = {
            atom.GetProp('tracking_idx'): (atom, Counter(
                [
                    (
                        bond.GetBondTypeAsDouble(),
                        bond.GetOtherAtom(atom).GetSymbol()
                    )
                    for bond in atom.GetBonds()
                ]
            ))
        for atom in mol.GetAtoms()
        }

        for atom in self.mol.GetAtoms():
            assert atom.GetProp('tracking_idx') in dict_mol_env.keys(), f"Error in tag_reactive_center: Tracking index {atom.GetProp('tracking_idx')} not found in reference molecule. Either the molecules are not aligned or the tracking index is not set properly."
            environment = Counter(
                [
                    (
                        bond.GetBondTypeAsDouble(),
                        bond.GetOtherAtom(atom).GetSymbol()
                    )
                    for bond in atom.GetBonds()
                ]
            )
            #print(f"Environment: {environment}")
            #print(f"Reference: {dict_mol_env[atom.GetProp('tracking_idx')]}")
            if dict_mol_env[atom.GetProp('tracking_idx')][1] != environment:
                atom.SetProp('reactive_center', '1')
                dict_mol_env[atom.GetProp('tracking_idx')][0].SetProp('reactive_center', '1')
            else:
                atom.SetProp('reactive_center', '0')
                dict_mol_env[atom.GetProp('tracking_idx')][0].SetProp('reactive_center', '0')

    def align_oneself_with(self, tracked_mol):
        canonical_idx_self = list(
            Chem.CanonicalRankAtoms(self.mol, breakTies=True)
        )
        canonical_idx_tracked = list(
            Chem.CanonicalRankAtoms(tracked_mol, breakTies=True)
        )
        assert set(canonical_idx_self) == set(
            canonical_idx_tracked
        ), f"Error in align_oneself_with: {canonical_idx_self} != {canonical_idx_tracked}"
        # self idx -> idx in the tracked mol
        # (Careful, idx in the tracked molecule are not tracked_idx prop)
        self_idx_to_tracked = {
            self_idx: canonical_idx_tracked.index(canonical_idx_self[self_idx])
            for self_idx in range(len(canonical_idx_self))
        }
        for self_idx, self_atom in enumerate(self.mol.GetAtoms()):
            tracked_atom = tracked_mol.GetAtomWithIdx(
                self_idx_to_tracked[self_idx]
            )
            self_atom.SetProp(
                "tracking_idx", tracked_atom.GetProp("tracking_idx")
            )

    def make_ionizing_move(self, bond_idx, atom_idx, track_atoms=False):

        if track_atoms and not self.mol.GetAtomWithIdx(0).HasProp(
            "tracking_idx"
        ):
            for idx, a in enumerate(self.mol.GetAtoms()):
                a.SetProp("tracking_idx", str(idx+1))

        mol = Chem.EditableMol(self.mol)

        bond = mol.GetMol().GetBondWithIdx(bond_idx)
        bond_degree = bond.GetBondTypeAsDouble()
        a_neg, a_pos = (
            (bond.GetBeginAtom(), bond.GetEndAtom())
            if (atom_idx == bond.GetBeginAtomIdx())
            else (bond.GetEndAtom(), bond.GetBeginAtom())
        )
        a_neg.SetFormalCharge(a_neg.GetFormalCharge() - 1)
        a_pos.SetFormalCharge(a_pos.GetFormalCharge() + 1)

        if bond_degree > 1:
            bond.SetBondType(MoleculeSet.dict_bond_type[bond_degree - 1])
            mol.ReplaceBond(bond_idx, bond)
        else:
            mol.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        mol.ReplaceAtom(a_neg.GetIdx(), a_neg)
        mol.ReplaceAtom(a_pos.GetIdx(), a_pos)
        return mol

    def make_attacking_move(self, lp_atom_idx, eo_atom_idx, track_atoms=False):
        # lp = lone pair, eo = empty orbital

        if track_atoms and not self.mol.GetAtomWithIdx(0).HasProp(
            "tracking_idx"
        ):
            for idx, a in enumerate(self.mol.GetAtoms()):
                a.SetProp("tracking_idx", str(idx))

        mol = Chem.EditableMol(self.mol)

        lp_atom = mol.GetMol().GetAtomWithIdx(lp_atom_idx)
        eo_atom = mol.GetMol().GetAtomWithIdx(eo_atom_idx)
        lp_atom.SetFormalCharge(lp_atom.GetFormalCharge() + 1)
        eo_atom.SetFormalCharge(eo_atom.GetFormalCharge() - 1)
        bond = mol.GetMol().GetBondBetweenAtoms(
            lp_atom.GetIdx(), eo_atom.GetIdx()
        )

        if bond is None:
            mol.AddBond(
                lp_atom.GetIdx(), eo_atom.GetIdx(), Chem.BondType.SINGLE
            )
        else:
            bond_degree = bond.GetBondTypeAsDouble()
            mol.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            mol.AddBond(
                lp_atom.GetIdx(),
                eo_atom.GetIdx(),
                MoleculeSet.dict_bond_type[bond_degree + 1],
            )
        mol.ReplaceAtom(lp_atom.GetIdx(), lp_atom)
        mol.ReplaceAtom(eo_atom.GetIdx(), eo_atom)
        return mol

    def clean_smiles_artefacts(self, smiles, verbose=False):
        # This function acts as a SMILES sanitizer, to avoid some simplification of SMILES that don't apply to this exact use case
        smiles_wo_aretacts_h = self.remove_artefact_hydrogens(smiles)
        cleaned_smiles = self.put_atoms_in_brackets(smiles_wo_aretacts_h)

        if verbose:
            print(
                f"clean_canonical_smiles: {smiles} -> {smiles_wo_aretacts_h} -> {cleaned_smiles}"
            )
        return cleaned_smiles

    def remove_artefact_hydrogens(self, smiles):
        # This function counteracts the addition that sometimes happens of
        # hydrogens that are neither implicit nor explicit but appear in the SMILES

        # This pattern eliminates any hydrogens not alone in square brackets
        pattern = r"(?<!\[)H\d*"
        cleaned_smiles = re.sub(pattern, "", smiles)
        return cleaned_smiles

    def put_atoms_in_brackets(self, smiles):
        # Put every atom in bracket, to avoid the apparition of implicit hydrogens

        # Match any atom symbol that doesn't need brackets in SMILES notation
        pattern = r"(?<!\[)(Br|B|Cl|C|N|O|P|S|F|I)"

        # Replace each match with the element surrounded by square brackets
        cleaned_smiles = re.sub(pattern, r"[\1]", smiles)

        return cleaned_smiles

    def atom_dictionary(self):
        # Return a dictionary with the number of each type of atom as well as the total charge.
        # This dictionary should always remain constant before and after a legal move, so this
        # function serves as an assertion test.
        atom_dict = defaultdict(int)

        for a in self.mol.GetAtoms():
            atom_dict[a.GetSymbol()] += 1
            atom_dict["charge"] += a.GetFormalCharge()

        return atom_dict

    def largest_common_substructure(self, other_ms):
        MCS_result = rdFMCS.FindMCS([self.mol, other_ms.mol])
        return MCS_result

    def tanimoto_similarity(self, other_ms):
        return DataStructs.TanimotoSimilarity(self.fp, other_ms.fp)

    def has_reached_goal(self, goal_ms):
        if self.smiles == goal_ms.smiles:
            return True
        # else
        try:
            can_smi = Chem.MolToSmiles(self.mol)
            goal_can_smi = Chem.MolToSmiles(goal_ms.mol)
        except:
            print(
                Fore.RED
                + f"Error in has_reached_goal with {self.smiles} and {goal_ms.smiles}{Fore.RESET}"
            )
            return False

        can_smi_list = can_smi.split(".")
        goal_can_smi_list = goal_can_smi.split(".")

        if len(can_smi_list) < len(goal_can_smi_list):
            return False

        for gs in goal_can_smi_list:
            if gs not in can_smi_list:
                return False
            else:
                can_smi_list.remove(gs)
        return True
