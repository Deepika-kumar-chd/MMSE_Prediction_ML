import sys
import numpy as np 
import pandas as pd

from sklearn.impute import SimpleImputer,KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from MMSE.constant.training_pipeline import TARGET_COLUMN
from MMSE.entity.artifact_entity import (DataTransformationArtifact, DataValidationArtifact)
from MMSE.entity.config_entity import DataTransformationConfig

from MMSE.exception import PredException
from MMSE.logger import logging
from MMSE.utils.main_utils import save_numpy_array_data,save_object



class DataTransformation:

    def __init__(self,data_validation_artifact:DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig):
        
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise PredException(e,sys)


    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise PredException(e,sys)

    @classmethod
    def get_data_transformer_object(cls) -> Pipeline:
        try:

            categorical_cols = ['APOE4']
            clinical_cols = ['CDRSB_bl','ADAS11_bl', 'ADAS13_bl', 'RAVLT_immediate_bl', 'RAVLT_learning_bl','RAVLT_forgetting_bl', 'RAVLT_perc_forgetting_bl', 'FAQ_bl']
            imaging_cols = ['Ventricles_bl', 'Hippocampus_bl', 'WholeBrain_bl','Entorhinal_bl', 'Fusiform_bl', 'MidTemp_bl']
            age_col = ['AGE']

            # Categorical pipeline
            categorical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore'))
            ])

            # Clinical scores pipeline
            clinical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            # Imaging biomarkers pipeline
            imaging_pipeline = Pipeline(steps=[
                ('imputer', KNNImputer(n_neighbors=5)),
                ('scaler', StandardScaler())
            ])

            # Age pipeline
            age_pipeline = Pipeline(steps=[
                ('scaler', StandardScaler())
            ])

            # Combine everything into a single ColumnTransformer
            preprocessor = ColumnTransformer(transformers=[
                ('cat', categorical_pipeline, categorical_cols),
                ('clinical', clinical_pipeline, clinical_cols),
                ('imaging', imaging_pipeline, imaging_cols),
                ('age', age_pipeline, age_col)
            ])

            return preprocessor
            
        except Exception as e:
            raise PredException(e,sys)


    def initiate_data_transformation(self) -> DataTransformationArtifact:

        try:
            train_df = DataTransformation.read_data(
                self.data_validation_artifact.valid_train_file_path
            )
            test_df = DataTransformation.read_data(
                self.data_validation_artifact.valid_test_file_path
            )
            preprocessor = self.get_data_transformer_object()

            #training dataframe
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN],axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]


            #testing dataframe
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN],axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]

            #fit and transform
            preprocessor_object= preprocessor.fit(input_feature_train_df)

            transformed_input_train_feature = preprocessor_object.transform(input_feature_train_df)
            transformed_input_test_feature = preprocessor_object.transform(input_feature_test_df)


            #concatenate features and target
            train_arr = np.c_[transformed_input_train_feature, np.array(target_feature_train_df)]
            test_arr = np.c_[transformed_input_test_feature, np.array(target_feature_test_df)]

            #save numpy array data 
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path,
                                  array= train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path,
                                  array= test_arr)
            save_object(self.data_transformation_config.transformed_object_file_path,
                        preprocessor_object)
            
            #preparing artifact
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path = self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path = self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path = self.data_transformation_config.transformed_test_file_path
            )
            
            logging.info(f"Data transformation artifact: {data_transformation_artifact}")

            return data_transformation_artifact
        except Exception as e:
            raise PredException(e,sys)