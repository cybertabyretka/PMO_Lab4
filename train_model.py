import numpy as np

import pandas as pd

from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso

import mlflow
from mlflow.models import infer_signature


def transform_dataset(dataset):
    x, y = dataset.drop(columns = ['price']), dataset['price']
    scaler = StandardScaler()
    power_trans = PowerTransformer()
    X_scale = scaler.fit_transform(x.values)

    Y_scale = power_trans.fit_transform(y.values.reshape(-1,1))

    return X_scale, Y_scale, power_trans


def eval_metrics(actual, prediction):
    rmse = np.sqrt(mean_squared_error(actual, prediction))
    mae = mean_absolute_error(actual, prediction)
    r2 = r2_score(actual, prediction)
    return rmse, mae, r2


def train():
    df = pd.read_csv("cars.csv")
    X, Y, power_transform = transform_dataset(df)
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y,
                                                      test_size=0.3,
                                                      random_state=42)

    lasso_parameters = {
        'max_iter': [1000, 2000, 5000, 7000, 10000],
        'alpha': [1e-6, 1e-5, 1e-4, 1e-2, 1],
        'random_state': [42]
    }

    mlflow.set_experiment("linear model cars")
    with mlflow.start_run():
        lasso = Lasso()
        clf = GridSearchCV(lasso, lasso_parameters, cv=3, n_jobs=4)
        clf.fit(X_train, Y_train.reshape(-1))
        best = clf.best_estimator_
        y_pred = best.predict(X_val)
        y_price_pred = power_transform.inverse_transform(y_pred.reshape(-1, 1))
        (rmse, mae, r2) = eval_metrics(power_transform.inverse_transform(Y_val), y_price_pred)

        final_parameters = best.get_params()
        for parameter in final_parameters:
            mlflow.log_param(parameter, final_parameters[parameter])

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        predictions = best.predict(X_train)
        signature = infer_signature(X_train, predictions)
        mlflow.sklearn.log_model(best, "model", signature=signature)

    runs = mlflow.search_runs()
    path2model = runs.sort_values("metrics.r2", ascending=False).iloc[0]['artifact_uri'].replace("file://", "") + '/model'
    print(path2model)
