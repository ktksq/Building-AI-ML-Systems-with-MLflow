# Copyright (C) 2025 Yuki Shiga
#
# This program is distributed under the terms of the GNU Affero General Public License version 3.
# For details, please refer to the LICENSE file.
import base64
import io
import logging
import os

import cv2
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from mlflow.models import set_model

# ロギング設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ReceiptDetectionModel(mlflow.pyfunc.PythonModel):
    """
    レシート検出のためのMLflow PythonModel
    YOLOWorldを使用して画像からレシートを検出します
    """

    def load_context(self, context):
        """
        モデルのロードと初期化を行います

        Args:
            context: MLflowのモデルコンテキスト
        """
        import sys

        logger.info(f"Python version: {sys.version}")

        # コンテキストがNoneの場合の対応
        if context is None:
            model_dir = os.getcwd()
            logger.info(f"Using model directory: {model_dir}")
        else:
            model_dir = os.path.dirname(context.artifacts["model_file"])
            logger.info(f"Model artifacts directory: {model_dir}")
            if os.path.exists(model_dir):
                logger.info(f"Model artifacts: {os.listdir(model_dir)}")

        # YOLOWorldモデルのロード
        try:
            # torchはCLIPが内部で使用するために必要
            import torch

            _ = torch.zeros(1)  # torchを使用して未使用警告を回避

            from ultralytics import YOLOWorld

            model_path = os.path.join(model_dir, "yolov8s-world.pt")
            if not os.path.exists(model_path):
                logger.warning(
                    f"Model file not found at {model_path}, downloading from pretrained..."
                )
                self.model = YOLOWorld("yolov8s-world.pt")
            else:
                logger.info(f"Loading model from {model_path}")
                self.model = YOLOWorld(model_path)

            # 検出対象クラスの設定
            self.model.set_classes(
                [
                    "paper",
                    "document",
                    "book",
                    "receipt",
                    "invoice",
                    "ticket",
                    "white area",
                ]
            )
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def _load_image_from_binary(self, image_binary):
        """
        バイナリデータから画像を読み込みます

        Args:
            image_binary: 画像のバイナリデータ（バイト列またはbase64エンコードされた文字列）

        Returns:
            NumPy配列として読み込まれた画像
        """
        try:
            # Serving経由だとstr型で渡されることがあるため、bytesに変換
            if isinstance(image_binary, str):
                image_binary = base64.b64decode(image_binary)

            # バイナリデータをBytesIOオブジェクトに変換
            image_file = io.BytesIO(image_binary)

            # バイトデータをNumPy配列に変換
            image_array = np.frombuffer(image_file.getvalue(), dtype=np.uint8)

            # OpenCVで画像としてデコード
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            if image is None:
                raise ValueError("Failed to decode image from binary data")

            return image
        except Exception as e:
            logger.error(f"Error loading image from binary: {str(e)}")
            raise

    def predict(self, context, model_input):
        """
        画像からレシートを検出します

        Args:
            context: MLflowのモデルコンテキスト
            model_input: 入力データ。pandas DataFrameで以下のカラムを想定:
                - image_data: 画像データ (バイナリデータまたはbase64エンコードされた文字列)
                - image_path: 画像ファイルのパス (文字列) ※image_dataがある場合は無視
                - conf_threshold: 信頼度閾値 (float、デフォルト: 0.1)

        Returns:
            pandas DataFrame: 検出結果。以下のカラムを含む:
                - boxes: 検出されたバウンディングボックスのリスト
                - confidence: 各バウンディングボックスの信頼度のリスト
        """
        logger.info("Starting prediction: model_input=%s", model_input)

        # DataFrameでない場合は変換
        if not isinstance(model_input, pd.DataFrame):
            if isinstance(model_input, dict):
                model_input = pd.DataFrame([model_input])
            else:
                raise ValueError("Input must be a DataFrame or a dictionary")

        results = []

        # 各行を処理
        for _, row in model_input.iterrows():
            # 画像の取得
            image = None
            if "image_data" in row and row["image_data"] is not None:
                image_data = row["image_data"]
                # NumPy配列として直接画像データが提供されている場合
                if isinstance(image_data, np.ndarray):
                    image = image_data
                # バイナリデータまたはbase64エンコードされた文字列の場合
                elif isinstance(image_data, (bytes, str)):
                    image = self._load_image_from_binary(image_data)
                else:
                    raise ValueError("Unsupported image_data format")
            elif "image_path" in row and row["image_path"] is not None:
                # 画像ファイルのパスが提供されている場合
                image_path = row["image_path"]
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image file not found: {image_path}")
                image = cv2.imread(image_path)
            else:
                raise ValueError("Either 'image_path' or 'image_data' must be provided")

            # 信頼度閾値の取得
            conf_threshold = row.get("conf_threshold", 0.1)

            # レシート検出の実行
            try:
                yolo_results = self.model.predict(
                    image,
                    conf=conf_threshold,
                )

                # 結果の解析
                detected_boxes = []
                confidence_scores = []

                if len(yolo_results) > 0 and len(yolo_results[0].boxes) > 0:
                    for i, box in enumerate(yolo_results[0].boxes.xywh):
                        x, y, w, h = box.tolist()
                        detected_boxes.append([x, y, w, h])

                        # 信頼度スコアの取得
                        if hasattr(yolo_results[0].boxes, "conf"):
                            confidence_scores.append(
                                float(yolo_results[0].boxes.conf[i])
                            )
                        else:
                            confidence_scores.append(0.0)  # デフォルト値

                # 結果の準備
                result = {
                    "boxes": detected_boxes,
                    "confidence": confidence_scores,
                }

                results.append(result)
            except Exception as e:
                logger.error(f"Error during prediction: {str(e)}")
                raise

        # 結果をDataFrameとして返す
        return pd.DataFrame(results)

    def _draw_bounding_boxes(
        self, image, boxes, confidence_scores=None, class_names=None
    ):
        """
        バウンディングボックスを画像に描画します

        Args:
            image: 元の画像
            boxes: バウンディングボックスのリスト [[x, y, w, h], ...]
            confidence_scores: 信頼度スコアのリスト (オプション)
            class_names: クラス名のリスト (オプション)

        Returns:
            バウンディングボックスが描画された画像
        """
        result_image = image.copy()
        for i, box in enumerate(boxes):
            xc, yc, w, h = map(int, box)
            x1, y1, x2, y2 = map(int, [xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2])

            # 画像の範囲内に収める
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.shape[1], x2)
            y2 = min(image.shape[0], y2)

            # バウンディングボックスの描画
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # ラベルテキストの準備
            label_text = ""
            if class_names is not None and i < len(class_names):
                label_text += class_names[i]
            if confidence_scores is not None and i < len(confidence_scores):
                label_text += f" {confidence_scores[i]:.2f}"

            # ラベルの描画
            if label_text:
                cv2.putText(
                    result_image,
                    label_text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

        return result_image


set_model(ReceiptDetectionModel())


if __name__ == "__main__":
    import base64
    import os

    # テスト用画像パスの確認（sample/image.jpgを使用）
    test_image_path = os.path.join(os.path.dirname(__file__), "sample/image.jpg")
    if os.path.exists(test_image_path):
        print(f"テスト画像が見つかりました: {test_image_path}")

        # 通常のパスを使用したテスト
        print("1. 画像パスを使用したテスト:")
        model = ReceiptDetectionModel()
        model.load_context(None)  # コンテキストのロード

        import pandas as pd

        input_df = pd.DataFrame(
            [
                {
                    "image_path": test_image_path,
                    "conf_threshold": 0.2,
                }
            ]
        )

        result = model.predict(None, input_df)
        print(f"検出結果: {len(result['boxes'][0])} 個のレシートを検出")

        # 画像をbase64エンコードしてバイナリデータとして渡すテスト
        print("\n2. base64エンコードした画像データを使用したテスト:")
        # 画像ファイルを読み込み
        with open(test_image_path, "rb") as image_file:
            image_binary = image_file.read()

        # base64エンコード
        image_base64 = base64.b64encode(image_binary).decode("utf-8")

        # base64エンコードされた画像データを使用してモデルを実行
        input_df_base64 = pd.DataFrame(
            [
                {
                    "image_data": image_base64,  # base64エンコードされた文字列
                    "conf_threshold": 0.2,
                }
            ]
        )

        result_base64 = model.predict(None, input_df_base64)
        print(f"検出結果 (base64): {len(result_base64['boxes'][0])} 個のレシートを検出")

        # 直接バイナリデータを使用してモデルを実行
        print("\n3. バイナリデータを直接使用したテスト:")
        input_df_binary = pd.DataFrame(
            [
                {
                    "image_data": image_binary,  # バイナリデータ
                    "conf_threshold": 0.2,
                }
            ]
        )

        result_binary = model.predict(None, input_df_binary)
        print(
            f"検出結果 (バイナリ): {len(result_binary['boxes'][0])} 個のレシートを検出"
        )
        print(result_binary)
    else:
        print(f"テスト画像が見つかりません: {test_image_path}")
    pass
