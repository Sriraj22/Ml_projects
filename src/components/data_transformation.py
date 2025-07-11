import os
import sys
import pandas as pd
import numpy as np

from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline

from src.exception_handling import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass
class DataTransforamtionConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransforamtionConfig()
    
    def get_data_tranformer_obj(self):
        '''this funciton is to just create 
        pickle files responsible 
        for converting cat to numerical or std scaler'''
        '''
        This function si responsible for data transformation
        
        '''
        try:
            numerical_columns = ["writing_score", "reading_score"]

            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline = Pipeline(
                steps= [
                    ('imputer',SimpleImputer(strategy="median")),
                    ('scaler', StandardScaler(with_mean=False)),
                ]
            )

            cat_pipeline = Pipeline(
                steps = [
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False)),
                   ]
            )
            logging.info("Numerical columns std scaling completed")
            logging.info("Categorical columns encoding completed")

# transformers : list of tuples --> List of (name, transformer, columns)
            preprocessor = ColumnTransformer(
                transformers=[
                ("numerical pipeline",num_pipeline,numerical_columns),
                ("categorical pipeline", cat_pipeline, categorical_columns)
            ])

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info ("Read train and test data completed")
            logging.info("Obtaining preprocessor object ")

            preprocessor_obj = self.get_data_tranformer_obj()

            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"]

            input_feature_train_df  =  train_df.drop([target_column_name], axis=1)
            target_feature_train_df =  train_df[target_column_name]

            input_feature_test_df   =  test_df.drop([target_column_name], axis=1)
            target_feature_test_df  =  test_df[target_column_name]

            logging.info("applying transformation on training and test df's ")

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr  = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
 
            logging.info("saving preprocessor object")
            
            '''
            this save object will be in utils 
            since its a common fucntionality can be used in entire project 
            '''

            save_object(file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj)

            return (train_arr,
                    test_arr, 
                    self.data_transformation_config.preprocessor_obj_file_path,
                    )



        except Exception as e:
            raise CustomException(e,sys)



