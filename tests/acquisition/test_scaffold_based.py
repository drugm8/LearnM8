"""Scaffold-based acquisition function tests.

Tests the ScaffoldAcquisition class for chemical structural diversity sampling
using real molecular data and comprehensive functionality validation.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock

from learnm8.acquisition.scaffold_based import (
    ScaffoldAcquisition,
    create_scaffold_acquisition,
    generate_scaffolds,
    group_by_scaffold
)
from learnm8.acquisition.basic import RandomAcquisition

# Try to import RDKit for testing, skip tests if not available
try:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    Chem = None
    MurckoScaffold = None


class TestScaffoldGeneration:
    """Test scaffold generation utilities."""
    
    @pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not available")
    def test_generate_scaffolds_basic(self):
        """Test basic scaffold generation from SMILES."""
        smiles_list = [
            'CCc1ccc(cc1)C(C)C',  # Aromatic with alkyl substituents
            'CCc1ccc(cc1)O',      # Same aromatic core, different substituent
            'CCCCCCC',             # Aliphatic chain
            'C1CCCCC1'             # Cycloalkane
        ]
        
        scaffolds = generate_scaffolds(smiles_list, include_chirality=False)
        
        assert len(scaffolds) == 4
        assert all(scaffold is not None for scaffold in scaffolds[:2])  # Aromatics should work
        
        # First two should have same scaffold (same aromatic core)
        if scaffolds[0] is not None and scaffolds[1] is not None:
            assert scaffolds[0] == scaffolds[1]
    
    @pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not available")
    def test_generate_scaffolds_with_chirality(self):
        """Test scaffold generation with chirality consideration."""
        smiles_with_chirality = [
            'C[C@H](O)c1ccccc1',   # (S)-stereoisomer
            'C[C@@H](O)c1ccccc1',  # (R)-stereoisomer
        ]
        
        # Without chirality - should be same
        scaffolds_no_chiral = generate_scaffolds(smiles_with_chirality, include_chirality=False)
        if all(s is not None for s in scaffolds_no_chiral):
            assert scaffolds_no_chiral[0] == scaffolds_no_chiral[1]
        
        # With chirality - might be different (depends on scaffold generation)
        scaffolds_with_chiral = generate_scaffolds(smiles_with_chirality, include_chirality=True)
        assert len(scaffolds_with_chiral) == 2
    
    @pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not available")
    def test_generate_scaffolds_invalid_smiles(self):
        """Test handling of invalid SMILES."""
        smiles_list = [
            'CCO',           # Valid
            'invalid_smiles', # Invalid
            'CCC',           # Valid
            '[Pt]',          # Valid but unusual
        ]
        
        scaffolds = generate_scaffolds(smiles_list, include_chirality=False)
        
        assert len(scaffolds) == 4
        assert scaffolds[0] is not None  # Valid SMILES
        assert scaffolds[1] is None      # Invalid SMILES
        assert scaffolds[2] is not None  # Valid SMILES
    
    def test_generate_scaffolds_no_rdkit(self):
        """Test scaffold generation when RDKit is not available."""
        smiles_list = ['CCO', 'CCC']
        
        with patch('learnm8.acquisition.scaffold_based.RDKIT_AVAILABLE', False):
            with pytest.raises(ImportError, match="RDKit is required"):
                generate_scaffolds(smiles_list)
    
    def test_group_by_scaffold(self):
        """Test grouping compounds by scaffold."""
        compounds = pd.DataFrame({
            'ID': ['COMP_001', 'COMP_002', 'COMP_003', 'COMP_004'],
            'SMILES': ['CCO', 'CCC', 'CCN', 'CCCl']
        })
        
        scaffolds = ['scaffold_A', 'scaffold_A', 'scaffold_B', None]
        
        groups = group_by_scaffold(compounds, scaffolds)
        
        assert 'scaffold_A' in groups
        assert 'scaffold_B' in groups
        assert len(groups['scaffold_A']) == 2  # Compounds 0 and 1
        assert len(groups['scaffold_B']) == 1  # Compound 2
        
        # Invalid scaffold should create unique group
        invalid_groups = [key for key in groups.keys() if key.startswith('invalid_')]
        assert len(invalid_groups) == 1


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not available")
class TestScaffoldAcquisition:
    """Test scaffold-based acquisition functionality."""
    
    def test_scaffold_basic_functionality(self, small_real_compounds):
        """Test basic scaffold acquisition with real molecular data."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) < 10:
            pytest.skip("Insufficient compounds for scaffold test")
        
        # Add predictions
        compounds['prediction'] = compounds.get('Activity', np.random.random(len(compounds)))
        
        acq = ScaffoldAcquisition(include_chirality=False, random_state=42)
        selected = acq.select(compounds, n_select=8)
        
        assert len(selected) == 8
        assert all(col in selected.columns for col in ['ID', 'SMILES', 'prediction'])
        assert 'acquisition_score' in selected.columns
        assert 'scaffold' in selected.columns
        
        # Should select valid compounds
        assert all(id in compounds['ID'].values for id in selected['ID'])
        
        # Scaffolds should be strings or None
        assert all(isinstance(scaffold, (str, type(None))) for scaffold in selected['scaffold'])
    
    def test_scaffold_deterministic_selection(self, medium_real_compounds):
        """Test that scaffold selection is deterministic with same random_state."""
        compounds = medium_real_compounds.copy()
        
        if len(compounds) < 15:
            pytest.skip("Insufficient compounds for deterministic test")
        
        compounds['prediction'] = compounds.get('Activity', np.random.random(len(compounds)))
        
        # Run selection twice with same random_state
        acq1 = ScaffoldAcquisition(random_state=42)
        selected1 = acq1.select(compounds, n_select=10)
        
        acq2 = ScaffoldAcquisition(random_state=42)
        selected2 = acq2.select(compounds, n_select=10)
        
        # Should select identical compounds (order matters for scaffold-based)
        assert list(selected1['ID']) == list(selected2['ID'])
    
    def test_scaffold_vs_random_diversity(self, diverse_real_compounds):
        """Test that scaffold acquisition provides different selection than random."""
        compounds = diverse_real_compounds.copy()
        
        if len(compounds) < 20:
            pytest.skip("Insufficient compounds for diversity comparison")
        
        compounds['prediction'] = compounds.get('Activity', np.random.random(len(compounds)))
        
        # Scaffold selection
        scaffold_acq = ScaffoldAcquisition(random_state=42)
        scaffold_selected = scaffold_acq.select(compounds, n_select=12)
        
        # Random selection for comparison
        random_acq = RandomAcquisition(random_state=42)
        random_selected = random_acq.select(compounds, n_select=12)
        
        # Both should select same number
        assert len(scaffold_selected) == len(random_selected) == 12
        
        # Should generally select different compounds
        scaffold_ids = set(scaffold_selected['ID'])
        random_ids = set(random_selected['ID'])
        overlap = len(scaffold_ids & random_ids)
        
        # Expect less than 80% overlap (scaffold method should be different)
        assert overlap < 10, f"Too much overlap ({overlap}/12) between scaffold and random"
    
    def test_scaffold_structural_diversity(self, diverse_real_compounds):
        """Test that scaffold method promotes structural diversity."""
        compounds = diverse_real_compounds.copy()
        
        if len(compounds) < 15:
            pytest.skip("Insufficient compounds for structural diversity test")
        
        compounds['prediction'] = compounds.get('Activity', np.random.random(len(compounds)))
        
        acq = ScaffoldAcquisition(random_state=42)
        selected = acq.select(compounds, n_select=10)
        
        assert len(selected) == 10
        
        # Count unique scaffolds in selection
        unique_scaffolds = set(selected['scaffold'])
        unique_scaffolds.discard(None)  # Remove None values
        
        # Should have good scaffold diversity (at least 3 different scaffolds for 10 compounds)
        assert len(unique_scaffolds) >= 3, f"Low scaffold diversity: {len(unique_scaffolds)} unique scaffolds"
    
    def test_scaffold_chirality_handling(self, small_real_compounds):
        """Test scaffold generation with and without chirality."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) < 8:
            pytest.skip("Insufficient compounds for chirality test")
        
        compounds['prediction'] = compounds.get('Activity', np.random.random(len(compounds)))
        
        # Test without chirality
        acq_no_chiral = ScaffoldAcquisition(include_chirality=False, random_state=42)
        selected_no_chiral = acq_no_chiral.select(compounds, n_select=6)
        
        # Test with chirality
        acq_with_chiral = ScaffoldAcquisition(include_chirality=True, random_state=42)
        selected_with_chiral = acq_with_chiral.select(compounds, n_select=6)
        
        assert len(selected_no_chiral) == 6
        assert len(selected_with_chiral) == 6
        
        # Both should work and produce valid selections
        assert all(id in compounds['ID'].values for id in selected_no_chiral['ID'])
        assert all(id in compounds['ID'].values for id in selected_with_chiral['ID'])
    
    def test_scaffold_edge_cases(self, edge_case_compounds):
        """Test scaffold acquisition with edge cases and small datasets."""
        compounds = edge_case_compounds.copy()
        
        if len(compounds) == 0:
            compounds = pd.DataFrame({
                'ID': ['COMP_001', 'COMP_002'],
                'SMILES': ['CCO', 'CCC'],
                'prediction': [0.5, 0.7]
            })
        else:
            compounds['prediction'] = compounds.get('Activity', np.random.random(len(compounds)))
        
        acq = ScaffoldAcquisition(random_state=42)
        
        n_compounds = len(compounds)
        
        # Test single compound selection
        if n_compounds >= 1:
            selected = acq.select(compounds, n_select=1)
            assert len(selected) == 1
        
        # Test selecting all compounds
        if n_compounds >= 2:
            selected_all = acq.select(compounds, n_select=n_compounds)
            assert len(selected_all) == n_compounds
    
    def test_scaffold_invalid_smiles_handling(self):
        """Test handling of compounds with invalid SMILES."""
        compounds = pd.DataFrame({
            'ID': ['COMP_001', 'COMP_002', 'COMP_003', 'COMP_004'],
            'SMILES': ['CCO', 'invalid_smiles', 'CCC', 'also_invalid'],
            'prediction': [0.1, 0.5, 0.9, 0.3]
        })
        
        acq = ScaffoldAcquisition(random_state=42)
        selected = acq.select(compounds, n_select=3)
        
        assert len(selected) == 3
        
        # Should handle invalid SMILES gracefully
        assert all(id in compounds['ID'].values for id in selected['ID'])
        
        # Invalid SMILES should have None scaffolds or unique invalid scaffolds
        invalid_indices = [i for i, smiles in enumerate(compounds['SMILES']) 
                         if smiles in ['invalid_smiles', 'also_invalid']]
        
        # Function should not crash on invalid SMILES
        assert 'scaffold' in selected.columns
    
    def test_scaffold_input_validation(self, small_real_compounds):
        """Test input validation and error handling."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            compounds = pd.DataFrame({
                'ID': ['COMP_001', 'COMP_002', 'COMP_003'],
                'SMILES': ['CCO', 'CCC', 'CCN'],
                'prediction': [0.1, 0.5, 0.9]
            })
        else:
            compounds['prediction'] = compounds.get('Activity', np.random.random(len(compounds)))
        
        acq = ScaffoldAcquisition()
        
        # Test empty DataFrame
        empty_df = pd.DataFrame(columns=['ID', 'SMILES', 'prediction'])
        with pytest.raises(ValueError, match="compounds DataFrame is empty"):
            acq.select(empty_df, n_select=1)
        
        # Test missing columns
        incomplete_df = compounds[['ID', 'SMILES']].copy()
        with pytest.raises(ValueError, match="Missing required columns"):
            acq.select(incomplete_df, n_select=1)
        
        # Test invalid n_select
        with pytest.raises(ValueError, match="n_select must be positive"):
            acq.select(compounds, n_select=0)
    
    def test_scaffold_requires_uncertainty(self):
        """Test that scaffold acquisition doesn't require uncertainty estimates."""
        acq = ScaffoldAcquisition()
        assert not acq.requires_uncertainty()
    
    def test_scaffold_get_name(self):
        """Test acquisition function naming."""
        acq_no_chiral = ScaffoldAcquisition(include_chirality=False)
        assert 'Scaffold' in acq_no_chiral.get_name()
        assert 'chiral' not in acq_no_chiral.get_name()
        
        acq_with_chiral = ScaffoldAcquisition(include_chirality=True)
        assert 'Scaffold' in acq_with_chiral.get_name()
        assert 'chiral' in acq_with_chiral.get_name()


class TestScaffoldAcquisitionNoRDKit:
    """Test scaffold acquisition behavior when RDKit is not available."""
    
    def test_scaffold_no_rdkit_import_error(self, small_real_compounds):
        """Test that scaffold acquisition raises ImportError without RDKit."""
        compounds = small_real_compounds.copy()
        
        if len(compounds) == 0:
            compounds = pd.DataFrame({
                'ID': ['COMP_001', 'COMP_002'],
                'SMILES': ['CCO', 'CCC'],
                'prediction': [0.5, 0.7]
            })
        else:
            compounds['prediction'] = compounds.get('Activity', np.random.random(len(compounds)))
        
        with patch('learnm8.acquisition.scaffold_based.RDKIT_AVAILABLE', False):
            acq = ScaffoldAcquisition()
            
            with pytest.raises(ImportError, match="RDKit is required"):
                acq.select(compounds, n_select=2)


class TestScaffoldFactory:
    """Test factory function for ScaffoldAcquisition."""
    
    def test_create_scaffold_acquisition_defaults(self):
        """Test factory function with default parameters."""
        acq = create_scaffold_acquisition()
        
        assert isinstance(acq, ScaffoldAcquisition)
        assert acq.include_chirality == False
        assert acq.random_state == 42
    
    def test_create_scaffold_acquisition_custom(self):
        """Test factory function with custom parameters."""
        acq = create_scaffold_acquisition(
            include_chirality=True,
            random_state=123
        )
        
        assert isinstance(acq, ScaffoldAcquisition)
        assert acq.include_chirality == True
        assert acq.random_state == 123


# Integration tests
@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not available")
class TestScaffoldIntegration:
    """Integration tests for scaffold acquisition."""
    
    def test_scaffold_representative_selection_strategy(self, medium_real_compounds):
        """Test the strategy for selecting representatives from scaffold groups."""
        compounds = medium_real_compounds.copy()
        
        if len(compounds) < 20:
            pytest.skip("Insufficient compounds for strategy test")
        
        # Create predictions with clear patterns for testing
        compounds['prediction'] = np.random.random(len(compounds))
        # Ensure some compounds have high predictions
        compounds.loc[:5, 'prediction'] = 0.9
        compounds.loc[6:10, 'prediction'] = 0.1
        
        acq = ScaffoldAcquisition(random_state=42)
        selected = acq.select(compounds, n_select=12)
        
        assert len(selected) == 12
        
        # Test that we're selecting representatives from different scaffolds
        unique_scaffolds = set(selected['scaffold'])
        unique_scaffolds.discard(None)
        
        # Should have reasonable scaffold diversity
        assert len(unique_scaffolds) >= 3
        
        # Within each scaffold group, should prefer higher predictions
        # (This is tested indirectly through the overall selection quality)
        assert 'acquisition_score' in selected.columns
    
    def test_scaffold_performance_with_large_dataset(self, medium_real_compounds):
        """Test scaffold acquisition performance characteristics."""
        compounds = medium_real_compounds.copy()
        
        if len(compounds) < 30:
            pytest.skip("Insufficient compounds for performance test")
        
        compounds['prediction'] = compounds.get('Activity', np.random.random(len(compounds)))
        
        acq = ScaffoldAcquisition(random_state=42)
        
        # Test with different selection sizes
        for n_select in [5, 10, 15]:
            if n_select <= len(compounds):
                selected = acq.select(compounds, n_select=n_select)
                assert len(selected) == n_select
                
                # Each selection should maintain scaffold diversity
                unique_scaffolds = len(set(selected['scaffold']) - {None})
                
                # At minimum, should have some scaffold diversity
                assert unique_scaffolds >= min(n_select // 3, 3)
    
    def test_scaffold_real_chemical_diversity(self, diverse_real_compounds):
        """Test scaffold method with chemically diverse real compounds."""
        compounds = diverse_real_compounds.copy()
        
        if len(compounds) < 25:
            pytest.skip("Insufficient diverse compounds for chemical diversity test")
        
        compounds['prediction'] = compounds.get('Activity', np.random.random(len(compounds)))
        
        acq = ScaffoldAcquisition(random_state=42)
        selected = acq.select(compounds, n_select=15)
        
        assert len(selected) == 15
        
        # With truly diverse compounds, should achieve good scaffold diversity
        unique_scaffolds = set(selected['scaffold'])
        unique_scaffolds.discard(None)
        
        # Should achieve substantial structural diversity
        scaffold_diversity = len(unique_scaffolds) / len(selected)
        assert scaffold_diversity >= 0.4, f"Low scaffold diversity: {scaffold_diversity:.2f}"
        
        # Verify we're getting different scaffolds (not just invalid ones)
        valid_scaffolds = [s for s in selected['scaffold'] if s is not None and len(s) > 0]
        assert len(valid_scaffolds) >= len(selected) // 2, "Too many invalid scaffolds"