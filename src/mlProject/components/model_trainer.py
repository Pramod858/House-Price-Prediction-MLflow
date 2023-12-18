import pandas as pd
import os
from src.mlProject import logger
from sklearn.ensemble import RandomForestRegressor
import joblib
from src.mlProject.entity.config_entity import ModelTrainerConfig



class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    
    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)


        X_train = train_data.drop(['price'], axis=1)
        X_test = test_data.drop(['price'], axis=1)
        y_train = train_data[['price']]
        y_test = test_data[['price']]


        lr = RandomForestRegressor(n_estimators=self.config.n_estimators, min_samples_split=self.config.min_samples_split, 
                                   min_samples_leaf=self.config.min_samples_leaf, max_features=self.config.max_features, random_state=42)
        lr.fit(X_train, y_train)

        joblib.dump(lr, os.path.join(self.config.root_dir, self.config.model_name))

