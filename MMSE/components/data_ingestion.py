from MMSE.exception import PredException
from MMSE.logger import logging
import os
import sys
from pandas import DataFrame
from MMSE.entity.config_entity import DataIngestionConfig
from MMSE.entity.artifact_entity import DataIngestionArtifact
from MMSE.data_access.data import Data
from sklearn.model_selection import train_test_split
from MMSE.utils.main_utils import read_yaml_file
from MMSE.constant.training_pipeline import SCHEMA_FILE_PATH

class DataIngestion:
    
    def __init__(self,data_ingestion_config:DataIngestionConfig):
        try:
            self.data_ingestion_config=data_ingestion_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise PredException(e,sys)

    def export_data_into_feature_store(self) -> DataFrame:
        ''' 
        export mongodb collection record as data frame into feature
        '''

        try:
            logging.info("Exporting data from mongodb to feature store")

            data = Data()

            dataframe = data.export_collection_as_dataframe(collection_name=self.data_ingestion_config.collection_name,
                                                            database_name=self.data_ingestion_config.database_name)

            feature_store_file_path = self.data_ingestion_config.feature_store_file_path

            #creating folders
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path,exist_ok=True)

            dataframe.to_csv(feature_store_file_path, index=False,header=True)

            return dataframe
        
        except Exception as e:
            raise PredException(e,sys)

    def split_data_as_train_test(self,dataframe:DataFrame) -> None:
        try:
            train_set, test_set = train_test_split(
                dataframe, test_size=self.data_ingestion_config.train_test_split_ratio
            )

            logging.info("Performed train test split on the dataframe")
            logging.info("Exited split_data_as_train_test method of Data_Ingestion class")

            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)

            logging.info(f"Exporting train and test file path")

            train_set.to_csv(
                self.data_ingestion_config.training_file_path, index=False, header=True
            )

            test_set.to_csv(
                self.data_ingestion_config.testing_file_path, index=False, header=True
            )

            logging.info(f"Exported train and test file path")

        except Exception as e:
            raise PredException(e,sys)
        
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            dataframe = self.export_data_into_feature_store()

            dataframe = dataframe.drop(self._schema_config["drop_columns"],axis=1)
            # Row filtering
            row_condition = self._schema_config.get("row_filter_condition")
            if row_condition:
                dataframe = dataframe.query(row_condition)

            # Drop duplicates
            drop_dup_config = self._schema_config.get("drop_duplicates")
            if drop_dup_config:
                subset = drop_dup_config.get("subset")
                keep = drop_dup_config.get("keep", "first")  # default to 'first'
                dataframe = dataframe.drop_duplicates(subset=subset, keep=keep)

            dataframe = dataframe.drop(self._schema_config["drop_columns_2"],axis=1)

            self.split_data_as_train_test(dataframe=dataframe)

            data_ingestion_artifact = DataIngestionArtifact(trained_file_path=self.
                data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path
            )

            return data_ingestion_artifact
        
        except Exception as e:
            raise PredException(e,sys)
        

    