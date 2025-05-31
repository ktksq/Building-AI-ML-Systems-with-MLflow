# Databricks notebook source
# Copyright (C) 2025 Yuki Shiga
#
# This program is distributed under the terms of the GNU Affero General Public License version 3.
# For details, please refer to the LICENSE file.

# COMMAND ----------

# COMMAND ----------
# MAGIC %pip install -U databricks-sdk==0.50.0
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC # レシート検出アプリのデプロイ
# MAGIC
# MAGIC このノートブックでは、Gradioで作成したレシート検出アプリをDatabricksアプリとしてデプロイします。

# COMMAND ----------

# MAGIC %md
# MAGIC ## 必要なライブラリのインポート

# COMMAND ----------

import json
import os
import time
from pathlib import Path

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.apps import (
    App,
    AppDeployment,
    AppResource,
    AppResourceServingEndpoint,
    AppResourceServingEndpointServingEndpointPermission,
)

# Databricksワークスペースクライアントの初期化
workspace = WorkspaceClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## アプリケーションの設定

# COMMAND ----------

# アプリの名前と説明
APP_NAME = "receipt-detection-app"
APP_DESCRIPTION = "レシート検出アプリケーション"

# モデルエンドポイント名（01_serving.pyで作成したエンドポイント）
MODEL_ENDPOINT_NAME = "shared_yuki_shiga_techbookfest_receipt_detection"

# アプリのソースディレクトリ
APP_SOURCE_DIR = os.getcwd() + "/apps"

# COMMAND ----------

# MAGIC %md
# MAGIC ## アプリケーションの存在確認

# COMMAND ----------

# 既存のアプリを確認
existing_app = None
try:
    existing_app = workspace.apps.get(name=APP_NAME)
    print(f"アプリ '{APP_NAME}' は既に存在します。")
    print(f"アプリの状態: {existing_app.app_status.state.value}")
except Exception as e:
    if "does not exist" in str(e):
        print(f"アプリ '{APP_NAME}' は存在しません。新規作成します。")
    else:
        raise e

# COMMAND ----------

# MAGIC %md
# MAGIC ## アプリケーションの作成

# COMMAND ----------

serving_endpoint = AppResourceServingEndpoint(
    name=MODEL_ENDPOINT_NAME,
    permission=AppResourceServingEndpointServingEndpointPermission.CAN_QUERY,
)
model_endpoint = AppResource(name="model_endpoint", serving_endpoint=serving_endpoint)

# アプリの作成
app = App(
    name=APP_NAME,
    description=APP_DESCRIPTION,
    resources=[model_endpoint],
)
# アプリが存在しない場合は作成
if existing_app is None:
    print(f"アプリ '{APP_NAME}' を作成しています...")

    # アプリを作成して完了を待つ
    created_app = workspace.apps.create_and_wait(app=app)

    print(f"アプリ '{APP_NAME}' を作成しました。")
    print(f"アプリID: {created_app.id}")
    print(f"アプリの状態: {created_app.app_status.state.value}")
else:
    print(f"既存のアプリ '{APP_NAME}' を使用します。")
    created_app = workspace.apps.update(name=APP_NAME, app=app)

# COMMAND ----------

# MAGIC %md
# MAGIC ## アプリケーションのデプロイ

# COMMAND ----------

# アプリのデプロイ設定
# AppDeploymentクラスを使用してデプロイ設定を指定
app_deployment = AppDeployment(source_code_path=APP_SOURCE_DIR)

# アプリをデプロイして完了を待つ
print(f"アプリ '{APP_NAME}' をデプロイしています...")
deployment = workspace.apps.deploy_and_wait(
    app_name=APP_NAME, app_deployment=app_deployment
)

print(f"アプリ '{APP_NAME}' のデプロイが完了しました。")
print(f"デプロイID: {deployment.deployment_id}")
print(f"デプロイの状態: {deployment.status.state.value}")

# COMMAND ----------
