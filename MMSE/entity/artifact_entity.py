from dataclasses import dataclass

@dataclass

class DataIngestionArtifact:
    trained_file_path:str
    test_file_path:str


@dataclass
class DataValidationArtifact:
    validation_status:str
    valid_train_file_path:str
    valid_test_file_path:str
    invalid_train_file_path:str
    invalid_test_file_path:str
    drift_report_file_path:str


@dataclass
class DataTransformationArtifact:
    transformed_object_file_path:str 
    transformed_train_file_path:str
    transformed_test_file_path:str


@dataclass
class MetricArtifact:
    R2: float
    RMSE: float
    MAE: float

@dataclass
class ModelTrainerArtifact:
    trained_model_file_path:str 
    trained_metric_artifact: MetricArtifact
    test_metric_artifact: MetricArtifact


@dataclass
class ModelPusherArtifact:
    saved_model_path:str
    model_file_path:str