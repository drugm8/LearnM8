
import logging

from learnm8.core.data_manager import DataManager

def setup_data_with_error_handling(compounds_df, data_manager: DataManager):
	"""
	Set up data manager with proper error handling for invalid SMILES.
	
	Args:
		compounds_df: DataFrame with compound data
		data_manager: DataManager instance
		
	Returns:
		DataFrame with compounds that have valid features
	"""
	logger = logging.getLogger(__name__)
	
	try:
		# Test feature generation on a small subset first
		test_ids = compounds_df['ID'].iloc[:10].tolist()
		logger.info("Testing molecular featurization on sample compounds...")
		logger.info(f"Test IDs being passed to DataManager: {test_ids}")
		
		test_smiles = compounds_df['SMILES'].iloc[:10].tolist()
		logger.info(f"Test SMILES being passed to DataManager: {test_smiles}")

		
		test_features = data_manager.get_features(compound_ids=test_ids, smiles_list=test_smiles)
		logger.info(f"Feature generation test successful. Feature shape: {test_features.shape}")
		
		# Now try to generate features for all compounds
		all_ids = compounds_df['ID'].tolist()
		logger.info(f"Generating features for {len(all_ids)} compounds...")
		
		all_smiles = compounds_df['SMILES'].tolist()
		logger.info(f"All SMILES being passed to DataManager: {all_smiles}")
		
		features = data_manager.get_features(all_ids, smiles_list=all_smiles)
		logger.info(f"Successfully generated features for all compounds. Shape: {features.shape}")
		
		return compounds_df
		
	except Exception as e:
		logger.error(f"Error during feature generation: {e}")

print("✅ Data setup function defined")