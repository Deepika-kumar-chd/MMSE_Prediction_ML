from MMSE.utils.main_utils import load_numpy_array_data
from MMSE.exception import PredException
from MMSE.logger import logging
from MMSE.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact
from MMSE.entity.config_entity import ModelTrainerConfig
import os,sys

from sklearn.linear_model import Ridge
from MMSE.ml.metric.metric import get_score
from MMSE.ml.model.estimator import PredModel
from MMSE.utils.main_utils import save_object,load_object


class ModelTrainer:

    def __init__(self,model_trainer_config:ModelTrainerConfig,data_transformation_artifact:DataTransformationArtifact):

        try:
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact


        except Exception as e:
            raise PredException(e,sys)


    def train_model(self,x_train,y_train):
        try:

            ridge_model = Ridge(alpha=10,max_iter=1000)
            ridge_model.fit(x_train, y_train)
            return ridge_model
        except Exception as e:
            raise PredException(e,sys)
        
    def initiate_model_trainer(self) ->ModelTrainerArtifact:

        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            #loading training array and testing array
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            x_train,y_train,x_test,y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1],
            )

            model = self.train_model(x_train,y_train)
            y_train_pred = model.predict(x_train)
            train_metric = get_score(y_true=y_train,y_pred=y_train_pred)
            logging.info(f"train metric: {train_metric}")
            if train_metric.R2 <= self.model_trainer_config.expected_R2:
                raise Exception("Trained model is not good to provide the expected R2")

            y_test_pred = model.predict(x_test)
            test_metric = get_score(y_true=y_test,y_pred= y_test_pred)
            logging.info(f"test metric: {test_metric}")


            preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)

            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path,exist_ok=True)
            pred_model = PredModel(preprocessor=preprocessor, model=model)
            save_object(self.model_trainer_config.trained_model_file_path, obj=pred_model)

            # model trainer artifact 

            model_trainer_artifact = ModelTrainerArtifact(trained_model_file_path = 
                    self.model_trainer_config.trained_model_file_path,
                    trained_metric_artifact = train_metric,
                    test_metric_artifact = test_metric,                                   
                )
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise PredException(e,sys)       
