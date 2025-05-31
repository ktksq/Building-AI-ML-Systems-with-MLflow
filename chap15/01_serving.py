# Databricks notebook source
# Copyright (C) 2025 Yuki Shiga
#
# This program is distributed under the terms of the GNU Affero General Public License version 3.
# For details, please refer to the LICENSE file.

# COMMAND ----------
# MAGIC %pip install -r model/requirements.txt
# MAGIC %pip install -U databricks-sdk==0.50.0
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC # レシート検出モデルのサービング
# MAGIC
# MAGIC このノートブックでは、ReceiptDetectionModelをMLflowに登録し、サービングのために準備します。

# COMMAND ----------

# MAGIC %md
# MAGIC ## 必要なライブラリのインポート

# COMMAND ----------

import os
import sys
import time

import mlflow
import mlflow.pyfunc
import pandas as pd
from databricks.sdk import WorkspaceClient
from databricks.sdk.runtime import *
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedModelInput

# Databricksワークスペースクライアントの初期化
workspace = WorkspaceClient()

# 現在のPythonバージョンを確認
print(f"Python version: {sys.version}")

# カタログとスキーマの設定
CATALOG = "shared"
SCHEMA = "yuki_shiga_techbookfest"

spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
spark.sql(f"USE SCHEMA {SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## モデルのインポートとテスト

# COMMAND ----------

# モデルのインポート
from model.reciept_detection import ReceiptDetectionModel

# モデルのインスタンス化
model = ReceiptDetectionModel()
model.load_context(None)  # コンテキストのロード

# COMMAND ----------

# MAGIC %md
# MAGIC ## サンプル画像でテスト実行

# COMMAND ----------

# サンプル画像のパスを設定
sample_image_path = os.path.join(os.getcwd(), "model/sample/image.jpg")

# ファイルの存在確認
if os.path.exists(sample_image_path):
    print(f"サンプル画像が見つかりました: {sample_image_path}")

    # テスト実行
    test_input = pd.DataFrame(
        [
            {
                "image_path": sample_image_path,
                "conf_threshold": 0.2,
            }
        ]
    )

    result = model.predict(None, test_input)

    print(f"検出結果: {len(result['boxes'][0])} 個のレシートを検出")
    print(f"検出されたバウンディングボックス: {result['boxes'][0]}")
    print(f"信頼度スコア: {result['confidence'][0]}")
else:
    print(f"サンプル画像が見つかりません: {sample_image_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## MLflowへのモデル登録

# COMMAND ----------

requirements = []
with open("model/requirements.txt") as f:
    for l in f.read().split("\n"):
        l = l.strip()
        if l == "" or l.startswith("#"):
            continue
        if "@" in l:
            l = l.split("@ ")[-1]
        requirements.append(l)
requirements

# COMMAND ----------

# MLflowでモデルをログ
mlflow.set_registry_uri("databricks-uc")

# モデルシグネチャの定義
import pandas as pd
from mlflow.models.signature import infer_signature

# 入力スキーマの定義
input_example = pd.DataFrame(
    [{"image_data": "base64encoded_string_here", "conf_threshold": 0.2}]
)

# 出力スキーマの定義
output_example = pd.DataFrame([{"boxes": [[100, 100, 200, 200]], "confidence": [0.95]}])

# シグネチャの作成
signature = infer_signature(input_example, output_example)

MODEL_NAME = f"{CATALOG}.{SCHEMA}.receipt_detection"
with mlflow.start_run(run_name="receipt_detection_model") as run:
    # タグの設定
    mlflow.set_tag("model_type", "receipt_detection")
    mlflow.set_tag("framework", "yoloworld")

    # パラメータの記録
    mlflow.log_param("yolo_model", "yolov8s-world.pt")
    mlflow.log_param("confidence_threshold", 0.2)

    # モデルを記録
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model="model/reciept_detection.py",
        pip_requirements=[
            "mlflow==2.21.3",
            "astunparse==1.6.3",
            "git+https://github.com/openai/CLIP.git",
            "cloudpickle==3.1.1",
            "defusedxml==0.7.1",
            "dill==0.3.6",
            "matplotlib==3.10.1",
            "numpy==2.2.5,>=1.20.0",
            "nvidia-ml-py==12.555.43",
            "optree==0.12.1",
            "pandas==2.2.3",
            "ultralytics==8.3.23,>=8.0.0",
            "opencv-python>=4.5.0",
            "torch>=1.7.1",
        ],
        registered_model_name=MODEL_NAME,
        artifacts={
            "model_file": "./yolov8s-world.pt",
        },
        signature=signature,
        input_example=input_example,
    )

    # 実行IDを取得
    run_id = run.info.run_id
    print(f"モデルを登録しました。Run ID: {run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## モデルのロードとテスト

# COMMAND ----------

import base64

# サンプル画像でテスト
if os.path.exists(sample_image_path):
    # 画像ファイルを読み込み
    with open(sample_image_path, "rb") as image_file:
        image_binary = image_file.read()

    # base64エンコード
    image_base64 = base64.b64encode(image_binary).decode("utf-8")

    # model load
    logged_model = f"runs:/{run.info.run_id}/model"
    loaded_model = mlflow.pyfunc.load_model(logged_model)

    # テスト実行
    test_input = pd.DataFrame(
        [
            {
                "image_data": image_base64,
                "conf_threshold": 0.2,
            }
        ]
    )

    loaded_result = loaded_model.predict(test_input)

    print(
        f"ロードしたモデルの検出結果: {len(loaded_result['boxes'][0])} 個のレシートを検出"
    )
    print(f"検出されたバウンディングボックス: {loaded_result['boxes'][0]}")
    print(f"信頼度スコア: {loaded_result['confidence'][0]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## サービングエンドポイントの作成
# MAGIC
# MAGIC 登録したモデルをサービングエンドポイントとしてデプロイします。
# MAGIC
# MAGIC 1. モデルの最新バージョンを取得
# MAGIC 2. エンドポイントの設定
# MAGIC 3. エンドポイントの作成または更新
# MAGIC 4. エンドポイントの状態確認

# COMMAND ----------

import mlflow
from databricks.sdk.service.serving import ServedModelInputWorkloadSize

# MLflowの設定
mlflow.set_registry_uri("databricks-uc")

# モデルの最新バージョンを取得
try:
    latest_version = (
        mlflow.MlflowClient()
        .search_model_versions(f"name='{MODEL_NAME}'", max_results=1)[0]
        .version
    )
    print(f"モデル '{MODEL_NAME}' の最新バージョン: {latest_version}")
except Exception as e:
    print(f"モデルバージョンの取得に失敗しました: {e}")
    print("最新のモデルを使用します")
    latest_version = "1"  # デフォルトバージョン

# エンドポイント名の設定（モデル名のドットをアンダースコアに置換）
endpoint_name = MODEL_NAME.replace(".", "_")
workload_size = ServedModelInputWorkloadSize.SMALL
scale_to_zero = True

# エンドポイントの設定
served_model = ServedModelInput(
    model_name=MODEL_NAME,
    model_version=latest_version,
    workload_size=workload_size,
    scale_to_zero_enabled=scale_to_zero,
)

endpoint_config = EndpointCoreConfigInput(
    served_models=[served_model],
)

# エンドポイントの存在確認
try:
    # エンドポイントが存在するか確認
    existing_endpoint = workspace.serving_endpoints.get(name=endpoint_name)
    print(f"エンドポイント '{endpoint_name}' は既に存在します。設定を更新します。")

    # エンドポイントの設定を更新
    workspace.serving_endpoints.update_config(
        name=endpoint_name,
        served_models=endpoint_config.served_models,
    )
except Exception as e:
    if "does not exist" in str(e):
        print(f"エンドポイント '{endpoint_name}' は存在しません。新規作成します。")
        # エンドポイントの作成
        workspace.serving_endpoints.create(name=endpoint_name, config=endpoint_config)
    else:
        raise e


# エンドポイントの状態を確認
def check_endpoint_status():
    endpoint = workspace.serving_endpoints.get(name=endpoint_name)
    print(f"エンドポイントの状態: {endpoint.state.ready}")
    return (
        endpoint.state.ready.value == "READY"
        and endpoint.state.config_update.value == "NOT_UPDATING"
    )


# エンドポイントの準備が完了するまで待機
print("エンドポイントの準備状態を確認中...")
while not check_endpoint_status():
    print("エンドポイントの準備中...")
    time.sleep(20)

print(f"エンドポイント '{endpoint_name}' の準備が完了しました。")

# エンドポイントの詳細を表示
endpoint_details = workspace.serving_endpoints.get(endpoint_name)
print(f"エンドポイントの詳細: {endpoint_details}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## モデルへのリクエスト送信
# MAGIC
# MAGIC WorkspaceClientを使用して、デプロイしたモデルにリクエストを送信します。

# COMMAND ----------

# サンプル画像をbase64エンコード
sample_image_path = os.path.join(os.getcwd(), "model/sample/image.jpg")

if os.path.exists(sample_image_path):
    import base64

    # 画像ファイルを読み込み
    with open(sample_image_path, "rb") as image_file:
        image_binary = image_file.read()

    # base64エンコード
    image_base64 = base64.b64encode(image_binary).decode("utf-8")

    # テスト用のデータを作成
    test_data = pd.DataFrame(
        [
            {
                "image_data": image_base64,
                "conf_threshold": 0.2,
            }
        ]
    )

    # モデルエンドポイントにリクエストを送信
    try:
        response = workspace.serving_endpoints.query(
            name=endpoint_name, dataframe_records=test_data.to_dict(orient="records")
        )

        # レスポンスを表示
        print("モデルからのレスポンス:")
        print(response.as_dict())
    except Exception as e:
        print(f"エンドポイントへのリクエスト送信中にエラーが発生しました: {e}")
else:
    print(f"サンプル画像が見つかりません: {sample_image_path}")
