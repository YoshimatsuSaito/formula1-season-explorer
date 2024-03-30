import os
from io import BytesIO

import boto3
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRanker


class Classifier:
    """Model to output winning probability of the driver"""

    def __init__(
        self,
        bucket_name: str,
        aws_access_key_id: str,
        aws_secret_access_key: str,
        s3_model_key: str,
    ) -> None:
        self.bucket_name = bucket_name
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.s3_model_key = s3_model_key
        self.s3 = boto3.client(
            "s3",
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
        )
        self.model = None

    def load_model(self, model_name: str = "classifier.pkl") -> None:
        """Load model from S3"""
        obj = self.s3.get_object(
            Bucket=self.bucket_name, Key=f"{self.s3_model_key}/{model_name}"
        )
        body = obj["Body"].read()
        model_stream = BytesIO(body)
        self.model = joblib.load(model_stream)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict winning probability"""
        return self.model.predict_proba(X)[:, 1]

    def train(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> None:
        """Train model"""
        self.model = LGBMClassifier(
            objective="binary",
            n_estimators=1000,
            metric="auc",
            class_weight="balanced",
        )
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(20, verbose=1), lgb.log_evaluation(period=0)],
        )

    def save_model(self, model_name: str = "classifier.pkl") -> None:
        """Save model to S3"""
        joblib.dump(self.model, model_name)
        self.s3.upload_file(
            Filename=model_name,
            Bucket=self.bucket_name,
            Key=f"{self.s3_model_key}/{model_name}",
        )
        os.remove(model_name)


class Ranker:
    """Model to predict the rank of the round for each driver"""

    def __init__(
        self,
        bucket_name: str,
        aws_access_key_id: str,
        aws_secret_access_key: str,
        s3_model_key: str,
    ) -> None:
        self.bucket_name = bucket_name
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.s3_model_key = s3_model_key
        self.s3 = boto3.client(
            "s3",
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
        )
        self.model = None

    def load_model(self, model_name: str = "ranker.pkl") -> None:
        """Load model from S3"""
        obj = self.s3.get_object(
            Bucket=self.bucket_name, Key=f"{self.s3_model_key}/{model_name}"
        )
        body = obj["Body"].read()
        model_stream = BytesIO(body)
        self.model = joblib.load(model_stream)

    def predict(self, X: pd.DataFrame, query: np.ndarray | None = None) -> np.ndarray:
        """Predict rank"""
        pred = self.model.predict(X)
        ranks = np.zeros_like(pred)

        # If prediction is only for a round, return the rank
        if query is None:
            # Start rank from 1
            return np.argsort(np.argsort(pred)) + 1

        # If prediction is for multiple rounds, return the rank for each round
        start_idx = 0
        for query_size in query:
            pred_query = pred[start_idx : start_idx + query_size]
            rank_query = np.argsort(np.argsort(pred_query))
            ranks[start_idx : start_idx + query_size] = rank_query
            start_idx += query_size
        # Start rank from 1
        return ranks + 1

    def train(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        query_train: np.ndarray,
        query_test: np.ndarray,
    ) -> None:
        """Train model"""
        self.model = LGBMRanker(
            objective="lambdarank",
            n_estimators=1000,
            metric="ndcg",
        )
        self.model.fit(
            X_train,
            y_train,
            group=query_train,
            eval_set=[(X_test, y_test)],
            eval_group=[query_test],
            eval_at=[1, 3, 10],
            callbacks=[lgb.early_stopping(20, verbose=1), lgb.log_evaluation(period=0)],
        )

    def save_model(self, model_name: str = "ranker.pkl") -> None:
        """Save model to S3"""
        joblib.dump(self.model, model_name)
        self.s3.upload_file(
            Filename=model_name,
            Bucket=self.bucket_name,
            Key=f"{self.s3_model_key}/{model_name}",
        )
        os.remove(model_name)
