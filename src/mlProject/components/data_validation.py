import os
from src.mlProject import logger
from src.mlProject.entity.config_entity import DataValidationConfig
import pandas as pd
from src.mlProject.utils.data_cleaning import DataCleaning, DataOutlierHandlingStrategy, DataPreprocessStrategy, DataCatToNumeric

class DataValiadtion:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def advanced_processing(self)-> bool:
        try:
            data = pd.read_csv(self.config.unzip_data_dir)

            outlier_stratergy = DataOutlierHandlingStrategy()
            data_cleaner = DataCleaning(data, outlier_stratergy)
            data_cleaned = data_cleaner.handle_data()
    
            preprocess_strategy = DataPreprocessStrategy()
            data_cleaning = DataCleaning(data_cleaned, preprocess_strategy)
            preprocessed_data = data_cleaning.handle_data()
    
            cat_to_numeric_strategy = DataCatToNumeric()
            data_cleaner = DataCleaning(preprocessed_data, cat_to_numeric_strategy)
            data_numeric = data_cleaner.handle_data()
            logger.info("Advanced pre processing is done")

            data_numeric.to_csv(self.config.preprocessed_data, index=False)

            logger.info("data file saved to given path")
            return True
        
        except Exception as e:
            raise e

