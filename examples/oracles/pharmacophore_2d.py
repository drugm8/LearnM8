import polars as pl
import numpy as np
from typing import List
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D
from learnm8.core.interfaces import Oracle
import logging

logger = logging.getLogger(__name__)


class Pharmacophore2DOracle(Oracle):

    def __init__(
        self,
        reference_smiles: str,
        metric: str = 'tanimoto'
    ):
        self.reference_smiles = reference_smiles
        self.metric = metric.lower()

        valid_metrics = ['tanimoto', 'dice']
        if self.metric not in valid_metrics:
            raise ValueError(f"metric must be one of {valid_metrics}")

        ref_mol = Chem.MolFromSmiles(reference_smiles)
        if ref_mol is None:
            raise ValueError(f"Invalid reference SMILES: {reference_smiles}")

        self.factory = Gobbi_Pharm2D.factory
        self.reference_fp = self._compute_pharmacophore_fp(ref_mol)

    def _compute_pharmacophore_fp(self, mol):
        try:
            return Generate.Gen2DFingerprint(mol, self.factory)
        except Exception as e:
            logger.warning(f"Failed to generate pharmacophore fingerprint: {e}")
            return None

    def _calculate_similarity(self, fp1, fp2):
        if fp1 is None or fp2 is None:
            return None

        if self.metric == 'tanimoto':
            return DataStructs.TanimotoSimilarity(fp1, fp2)
        elif self.metric == 'dice':
            return DataStructs.DiceSimilarity(fp1, fp2)
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

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

        similarities = []
        smiles_list = compounds['SMILES'].to_list()

        for smiles in smiles_list:
            try:
                if smiles is None or smiles == '':
                    similarities.append(None)
                    continue

                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    logger.warning(f"Invalid SMILES: {smiles}")
                    similarities.append(None)
                    continue

                fp = self._compute_pharmacophore_fp(mol)
                if fp is None:
                    similarities.append(None)
                    continue

                similarity = self._calculate_similarity(self.reference_fp, fp)
                similarities.append(similarity)

            except Exception as e:
                logger.warning(f"Failed to compute pharmacophore similarity for {smiles}: {e}")
                similarities.append(None)

        result = compounds.select('ID').with_row_index('_order')

        for prop in properties:
            result = result.with_columns(
                pl.Series(name=prop, values=similarities)
            )

        result = result.sort('_order').drop('_order')

        return result
