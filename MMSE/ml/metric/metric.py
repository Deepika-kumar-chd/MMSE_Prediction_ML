from MMSE.entity.artifact_entity import MetricArtifact
from MMSE.exception import PredException
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
import sys

def get_score(y_true,y_pred)->MetricArtifact:
    try:

        model_R2 = r2_score(y_true, y_pred)
        model_RMSE = np.sqrt(mean_squared_error(y_true, y_pred))
        model_MAE = mean_absolute_error(y_true, y_pred)

        regression_metric = MetricArtifact(
            R2 = model_R2,
            RMSE = model_RMSE,
            MAE = model_MAE
        )
        return regression_metric
    
    except Exception as e:
        raise PredException(e,sys)