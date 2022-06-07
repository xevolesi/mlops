import os
import pickle
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import mlflow
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from prefect import flow, task, get_run_logger
from prefect.task_runners import SequentialTaskRunner
from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule, IntervalSchedule
from prefect.flow_runners import SubprocessFlowRunner


@task
def get_paths(date: datetime = None) -> tuple[str, str]:
    if date is None:
        date = datetime.now()

    # Path to folder where this script lies.
    base_path = "./data"
    
    # Build path to training data.
    training_date = date - relativedelta(months=+2)
    training_data_file_name = f"fhv_tripdata_{training_date.year}-0{training_date.month}.parquet"
    training_data_path = os.path.join(base_path, training_data_file_name)

    # Build path to validation data.
    validation_date = date - relativedelta(months=+1)
    validation_data_file_name = f"fhv_tripdata_{validation_date.year}-0{validation_date.month}.parquet"
    validation_data_path = os.path.join(base_path, validation_data_file_name)
    return training_data_path, validation_data_path


@task
def read_data(path):
    df = pd.read_parquet(path)
    return df

@task
def prepare_features(df, categorical, train=True):
    logger = get_run_logger()
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        logger.info(f"The mean duration of training is {mean_duration}")
    else:
        logger.info(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

@task
def train_model(df, categorical):
    logger = get_run_logger()
    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts) 
    y_train = df.duration.values

    logger.info(f"The shape of X_train is {X_train.shape}")
    logger.info(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    mlflow.log_metric("Train MSE", mse)
    logger.info(f"The MSE of training is: {mse}")
    return lr, dv

@task
def run_model(df, categorical, dv, lr):
    logger = get_run_logger()
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    mlflow.log_metric("Validation MSE", mse)
    logger.info(f"The MSE of validation is: {mse}")
    return


@flow
def main(date: datetime = None):

    mlflow.set_tracking_uri("http://127.0.0.1:3000")
    mlflow.set_experiment("nyc-taxi-trip")
    mlflow.sklearn.autolog()

    categorical = ['PUlocationID', 'DOlocationID']
    train_path, val_path = get_paths(date).result()

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    with mlflow.start_run():
        lr, dv = train_model(df_train_processed, categorical).result()
        lr_model_name = f"models/model-{date.year}-{date.month}-{date.day}.bin"
        dv_model_name = f"models/dv-{date.year}-{date.month}-{date.day}.bin"
        pickle.dump(lr, open(lr_model_name, "wb"))
        pickle.dump(dv, open(dv_model_name, "wb"))
        run_model(df_val_processed, categorical, dv, lr)
        mlflow.log_artifact(dv_model_name)


DeploymentSpec(
    name="interval-deployment",
    flow=main,
    schedule=IntervalSchedule(interval=timedelta(minutes=3)),
    flow_runner=SubprocessFlowRunner()
)
