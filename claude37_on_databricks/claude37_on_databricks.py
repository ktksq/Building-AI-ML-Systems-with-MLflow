# Databricks notebook source
# MAGIC %md
# MAGIC # Databricks環境でのClaude 3.7 Sonnet実践ガイド：サンプルノートブック

# COMMAND ----------

# MAGIC %md
# MAGIC ## パッケージインストール

# COMMAND ----------

# MAGIC %pip install openai
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pythonリクエスト例

# COMMAND ----------

import json
import os
from openai import OpenAI

# トークンとワークスペースURLをノートブックコンテキストから取得
DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
WORKSPACE_URL = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()

# OpenAIクライアントの初期化
client = OpenAI(
    api_key=DATABRICKS_TOKEN,
    base_url=f"{WORKSPACE_URL}/serving-endpoints"
)

# DatabricksのClaude 3.7 Sonnetモデルを使用してリクエストを送信
response = client.chat.completions.create(
    model="databricks-claude-3-7-sonnet",
    messages=[{
        "role": "user",
        "content": "Databricksでのデータ分析ベストプラクティスを3つ教えてください。それぞれ一言ずつで、余分な装飾なしで出力してください。"
    }],
    max_tokens=500
)

# 結果を表示
print(response.choices[0].message.content)

# COMMAND ----------

# MAGIC %md
# MAGIC ## SQLリクエスト例

# COMMAND ----------

# MAGIC %sql
# MAGIC -- サンプルデータを含む一時ビューを作成
# MAGIC CREATE OR REPLACE TEMP VIEW customer_feedback AS
# MAGIC VALUES 
# MAGIC   ('C1001', '配送が予定より2日早く届いて驚きました。商品の品質も素晴らしく、また利用したいと思います。'),
# MAGIC   ('C1002', '注文から3週間経ちますが商品がまだ届きません。問い合わせにも返答がなく非常に困っています。'),
# MAGIC   ('C1003', '商品自体は期待通りでしたが、梱包が少し雑な印象を受けました。それ以外は特に問題ありません。')
# MAGIC AS t(customer_id, feedback_text);
# MAGIC
# MAGIC -- 顧客フィードバックの感情分析と要約を行うクエリ
# MAGIC WITH analysis_results AS (
# MAGIC   SELECT 
# MAGIC     customer_id,
# MAGIC     feedback_text,
# MAGIC     ai_query(
# MAGIC       'databricks-claude-3-7-sonnet',
# MAGIC       CONCAT('以下の顧客フィードバックを分析し、感情と要約をJSON形式で返してください。
# MAGIC       必ず {"sentiment": "ポジティブ/ネガティブ/ニュートラルのいずれか", "summary": "要約文"} という形式のJSONのみを返してください: ', feedback_text),
# MAGIC       modelParameters => named_struct(
# MAGIC         'max_tokens', 500
# MAGIC       )
# MAGIC     ) AS json_response
# MAGIC   FROM customer_feedback
# MAGIC )
# MAGIC -- 分析結果を整形して表示
# MAGIC SELECT 
# MAGIC   customer_id,
# MAGIC   feedback_text,
# MAGIC   get_json_object(json_response, '$.sentiment') AS `感情`,
# MAGIC   get_json_object(json_response, '$.summary') AS `要約`
# MAGIC FROM analysis_results;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pythonリクエスト例：拡張思考モード

# COMMAND ----------

import json
import os
from openai import OpenAI

# トークンとワークスペースURLをノートブックコンテキストから取得
DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
WORKSPACE_URL = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()

# OpenAIクライアントの初期化
client = OpenAI(
    api_key=DATABRICKS_TOKEN,
    base_url=f"{WORKSPACE_URL}/serving-endpoints"
)

# DatabricksのClaude 3.7 Sonnetモデルを使用してリクエストを送信：拡張思考モードを有効化
response = client.chat.completions.create(
    model="databricks-claude-3-7-sonnet",
    messages=[{
        "role": "user",
        "content": "二次方程式 2x² - 5x - 3 = 0 を解いてください。解だけを簡潔に答えてください。"
    }],
    max_tokens=8000,
    extra_body={
        "thinking": {
            "type": "enabled",
            "budget_tokens": 4000
        }
    }
)

# 結果をJSON形式で整形して表示
print(json.dumps(response.choices[0].message.content, indent=2, ensure_ascii=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## SQLリクエスト例：拡張思考モード
# MAGIC 現時点では拡張思考モード未サポートのエラーが出るためコメントアウト
# MAGIC ```
# MAGIC [AI_FUNCTION_INVALID_MODEL_PARAMETERS.UNSUPPORTED_MODEL_PARAMETER] The provided model parameters ({"max_tokens":8000,"thinking":{"type":"enabled","budget_tokens":4000}}) are invalid in the AI_QUERY function for serving endpoint "databricks-claude-3-7-sonnet". The "thinking" parameter is not supported for the serving endpoint. SQLSTATE: 22023
# MAGIC ```

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 拡張思考モードを利用した複雑な問題解決
# MAGIC -- SELECT ai_query(
# MAGIC --   'databricks-claude-3-7-sonnet',
# MAGIC --   '二次方程式 2x² - 5x - 3 = 0 を解いてください。',
# MAGIC --   modelParameters => named_struct(
# MAGIC --     'max_tokens', 8000,
# MAGIC --     'thinking', named_struct(
# MAGIC --       'type', 'enabled',
# MAGIC --       'budget_tokens', 4000
# MAGIC --     )
# MAGIC --   )
# MAGIC -- ) AS `二次方程式の解`;
