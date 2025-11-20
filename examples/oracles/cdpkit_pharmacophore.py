import polars as pl
import numpy as np
from typing import List, Optional
from rdkit import Chem
from rdkit.Chem import AllChem
from learnm8.core.interfaces import Oracle
import logging

logger = logging.getLogger(__name__)

try:
    import CDPL.Chem as CDPLChem
    import CDPL.Pharm as CDPLPharm
    import CDPL.Math as CDPLMath
    CDPKIT_AVAILABLE = True
except ImportError:
    CDPKIT_AVAILABLE = False
    logger.warning("CDPKit not available. Install with: pip install cdpkit")


class CDPKitPharmacophoreOracle(Oracle):

    def __init__(
        self,
        reference_smiles: str,
        score_type: str = 'alignment_score',
        exhaustive: bool = False,
        min_feature_match: int = 3,
        max_conformers: int = 10
    ):
        if not CDPKIT_AVAILABLE:
            raise ImportError("CDPKit is required for 3D pharmacophore calculations. Install with: pip install cdpkit")

        self.reference_smiles = reference_smiles
        self.score_type = score_type.lower()
        self.exhaustive = exhaustive
        self.min_feature_match = min_feature_match
        self.max_conformers = max_conformers

        valid_score_types = ['alignment_score', 'rmsd', 'feature_overlap']
        if self.score_type not in valid_score_types:
            raise ValueError(f"score_type must be one of {valid_score_types}")

        ref_mol_3d = self._generate_3d_conformer(reference_smiles)
        if ref_mol_3d is None:
            raise ValueError(f"Failed to generate 3D conformer for reference: {reference_smiles}")

        self.reference_pharmacophore = self._generate_pharmacophore(ref_mol_3d)
        if self.reference_pharmacophore is None:
            raise ValueError(f"Failed to generate pharmacophore for reference: {reference_smiles}")

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

    def _rdkit_to_cdpl(self, mol: Chem.Mol) -> Optional[CDPLChem.BasicMolecule]:
        try:
            mol_block = Chem.MolToMolBlock(mol)

            cdpl_mol = CDPLChem.BasicMolecule()
            reader = CDPLChem.MolecularGraphReader.fromString(mol_block, CDPLChem.MolecularGraphReader.Format.MOL)

            if not reader.read(cdpl_mol):
                return None

            return cdpl_mol

        except Exception as e:
            logger.warning(f"Failed to convert RDKit mol to CDPL: {e}")
            return None

    def _generate_pharmacophore(self, mol: Chem.Mol) -> Optional[CDPLPharm.BasicPharmacophore]:
        try:
            cdpl_mol = self._rdkit_to_cdpl(mol)
            if cdpl_mol is None:
                return None

            CDPLPharm.prepareForPharmacophoreGeneration(cdpl_mol)

            ph4_gen = CDPLPharm.DefaultPharmacophoreGenerator()
            ph4 = CDPLPharm.BasicPharmacophore()

            ph4_gen.generate(cdpl_mol, ph4)

            if ph4.getNumFeatures() < self.min_feature_match:
                logger.warning(f"Pharmacophore has only {ph4.getNumFeatures()} features (min: {self.min_feature_match})")
                return None

            return ph4

        except Exception as e:
            logger.warning(f"Failed to generate pharmacophore: {e}")
            return None

    def _align_pharmacophores(self, query_ph4: CDPLPharm.BasicPharmacophore) -> Optional[float]:
        try:
            aligner = CDPLPharm.PharmacophoreFitScreener(self.reference_pharmacophore)

            aligner.setCheckXVolumeClashes(True)
            aligner.setScoringFunction(CDPLPharm.PharmacophoreFitScreener.ScoringFunction.FEATURE_MATCH_COUNT)

            if query_ph4.getNumFeatures() < self.min_feature_match:
                return None

            score = aligner.screen(query_ph4)

            if score < 0:
                return None

            if self.score_type == 'alignment_score':
                return float(score)
            elif self.score_type == 'feature_overlap':
                max_features = max(self.reference_pharmacophore.getNumFeatures(), query_ph4.getNumFeatures())
                return float(score) / max_features if max_features > 0 else 0.0
            elif self.score_type == 'rmsd':
                return 1.0 / (1.0 + float(score))
            else:
                return float(score)

        except Exception as e:
            logger.warning(f"Failed to align pharmacophores: {e}")
            return None

    def _calculate_pharmacophore_score(self, smiles: str) -> Optional[float]:
        try:
            mol_3d = self._generate_3d_conformer(smiles)
            if mol_3d is None:
                return None

            query_ph4 = self._generate_pharmacophore(mol_3d)
            if query_ph4 is None:
                return None

            score = self._align_pharmacophores(query_ph4)
            return score

        except Exception as e:
            logger.warning(f"Failed to calculate pharmacophore score for {smiles}: {e}")
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

        scores = []
        smiles_list = compounds['SMILES'].to_list()

        for smiles in smiles_list:
            try:
                if smiles is None or smiles == '':
                    scores.append(None)
                    continue

                score = self._calculate_pharmacophore_score(smiles)
                scores.append(score)

            except Exception as e:
                logger.warning(f"Failed to compute pharmacophore score for {smiles}: {e}")
                scores.append(None)

        result = compounds.select('ID').with_row_index('_order')

        for prop in properties:
            result = result.with_columns(
                pl.Series(name=prop, values=scores)
            )

        result = result.sort('_order').drop('_order')

        return result
