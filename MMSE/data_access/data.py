import sys
from typing import Optional
import numpy as np 
import pandas as pd 
import json 
from MMSE.configuration.mongodb_db_connection import MongoDBClient
from MMSE.constant.database import DATABASE_NAME
from MMSE.exception import PredException
from MMSE.logger import logging


class Data:
    '''
    exports entire mongo db record as pandas dataframe 
    '''

    def __init__(self):
        
        try:
            self.mongo_client = MongoDBClient(database_name=DATABASE_NAME)

        except Exception as e:
            raise PredException(e,sys)


    def save_csv_file(self, file_path, collection_name:str, database_name: Optional[str] = None):
        try:
            data_frame=pd.read_csv(file_path)
            data_frame.reset_index(drop=True, inplace=True)
            records = list(json.loads(data_frame.T.to_json()).values())
            if database_name is None:
                collection = self.mongo_client.database[collection_name]
            else:
                collection = self.mongo_client[database_name][collection_name]
            collection.insert_many(records)
            return len(records)
        except Exception as e :
            raise PredException(e,sys)
        
    def export_collection_as_dataframe(self,collection_name:str,database_name:Optional[str]=None) -> pd.DataFrame:
        try:
            """
            export entire collection as dataframe:
            return pd.dataframe of collection
            """
            logging.info(f"Exporting collection: {collection_name}")
            logging.info(f"Exporting database: {database_name}")
            if database_name is None:
                collection = self.mongo_client.database[collection_name]
            else:
                collection = self.mongo_client.client[database_name][collection_name]
            df = pd.DataFrame(list(collection.find()))

            if "_id" in df.columns.to_list():
                df = df.drop(columns=["_id"],axis=1)
            # Drop the Unnamed index column if present
            if "Unnamed: 0" in df.columns:
                df.drop(columns=["Unnamed: 0"], inplace=True)

            df.replace({"na":np.nan}, inplace= True)
            logging.info(f"Exported dataframe shape: {df.shape}")
            return df

        except Exception as e :
            raise PredException(e,sys)