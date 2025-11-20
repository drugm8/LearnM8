import polars as pl
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from learnm8.core.interfaces import Oracle
import logging
import tempfile

logger = logging.getLogger(__name__)

try:
    from vina import Vina
    VINA_AVAILABLE = True
except ImportError:
    VINA_AVAILABLE = False
    logger.warning("Vina not available. Install with: conda install -c conda-forge vina")

try:
    from meeko import MoleculePreparation, PDBQTWriterLegacy
    MEEKO_AVAILABLE = True
except ImportError:
    MEEKO_AVAILABLE = False
    logger.warning("Meeko not available. Install with: conda install -c conda-forge meeko")


class VinaOracle(Oracle):

    def __init__(
        self,
        receptor_path: str,
        center: Tuple[float, float, float],
        box_size: Tuple[float, float, float] = (20.0, 20.0, 20.0),
        exhaustiveness: int = 8,
        n_poses: int = 1,
        energy_range: float = 3.0
    ):
        if not VINA_AVAILABLE:
            raise ImportError("Vina is required for docking. Install with: conda install -c conda-forge vina")

        if not MEEKO_AVAILABLE:
            raise ImportError("Meeko is required for PDBQT conversion. Install with: conda install -c conda-forge meeko")

        self.receptor_path = Path(receptor_path)
        if not self.receptor_path.exists():
            raise FileNotFoundError(f"Receptor file not found: {receptor_path}")

        self.center = center
        self.box_size = box_size
        self.exhaustiveness = exhaustiveness
        self.n_poses = n_poses
        self.energy_range = energy_range

        logger.info(f"Initialized VinaOracle with receptor: {receptor_path}")
        logger.info(f"Docking box center: {center}, size: {box_size}")

    def _generate_3d_conformer(self, smiles: str) -> Optional[Chem.Mol]:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            mol = Chem.AddHs(mol)

            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            params.numThreads = 1

            result = AllChem.EmbedMolecule(mol, params)
            if result == -1:
                logger.warning(f"Failed to embed molecule: {smiles}")
                return None

            AllChem.MMFFOptimizeMolecule(mol)

            return mol

        except Exception as e:
            logger.warning(f"Failed to generate 3D conformer for {smiles}: {e}")
            return None

    def _mol_to_pdbqt_string(self, mol: Chem.Mol) -> Optional[str]:
        try:
            preparator = MoleculePreparation()
            mol_setups = preparator.prepare(mol)

            if not mol_setups:
                logger.warning("Meeko preparation failed")
                return None

            pdbqt_string, is_ok, error_msg = PDBQTWriterLegacy.write_string(mol_setups[0])

            if not is_ok:
                logger.warning(f"PDBQT conversion error: {error_msg}")
                return None

            return pdbqt_string

        except Exception as e:
            logger.warning(f"Failed to convert molecule to PDBQT: {e}")
            return None

    def _dock_molecule(self, smiles: str) -> Optional[float]:
        try:
            mol_3d = self._generate_3d_conformer(smiles)
            if mol_3d is None:
                return None

            pdbqt_string = self._mol_to_pdbqt_string(mol_3d)
            if pdbqt_string is None:
                return None

            with tempfile.NamedTemporaryFile(mode='w', suffix='.pdbqt', delete=False) as f:
                f.write(pdbqt_string)
                ligand_path = f.name

            try:
                v = Vina(sf_name='vina', verbosity=0)

                v.set_receptor(str(self.receptor_path))

                v.set_ligand_from_file(ligand_path)

                v.compute_vina_maps(
                    center=list(self.center),
                    box_size=list(self.box_size)
                )

                v.dock(
                    exhaustiveness=self.exhaustiveness,
                    n_poses=self.n_poses
                )

                energies = v.energies(n_poses=1)

                if energies and len(energies) > 0:
                    best_affinity = energies[0][0]
                    return float(best_affinity)
                else:
                    return None

            finally:
                Path(ligand_path).unlink(missing_ok=True)

        except Exception as e:
            logger.warning(f"Docking failed for {smiles}: {e}")
            return None

    def measure(self, compounds: pl.DataFrame, properties: List[str]) -> pl.DataFrame:
        if 'SMILES' not in compounds.columns or 'ID' not in compounds.columns:
            raise ValueError("Compounds DataFrame must have 'ID' and 'SMILES' columns")

        if compounds.height == 0:
            result = pl.DataFrame({
                'ID': pl.Series([], dtype=pl.String)
            })
            for prop in properties:
                result = result.with_columns(pl.lit(None).cast(pl.Float64).alias(prop))
            return result

        affinities = []
        smiles_list = compounds['SMILES'].to_list()

        for i, smiles in enumerate(smiles_list):
            try:
                if smiles is None or smiles == '':
                    affinities.append(None)
                    continue

                logger.info(f"Docking compound {i+1}/{len(smiles_list)}: {smiles}")
                affinity = self._dock_molecule(smiles)
                affinities.append(affinity)

            except Exception as e:
                logger.warning(f"Failed to dock {smiles}: {e}")
                affinities.append(None)

        result = compounds.select('ID').with_row_index('_order')

        for prop in properties:
            result = result.with_columns(
                pl.Series(name=prop, values=affinities)
            )

        result = result.sort('_order').drop('_order')

        return result
