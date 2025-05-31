# Copyright (C) 2025 Yuki Shiga
#
# This program is distributed under the terms of the GNU Affero General Public License version 3.
# For details, please refer to the LICENSE file.m. If not, see <https://www.gnu.org/licenses/>.
import base64
import os
import tempfile
import zipfile
from typing import List, Tuple

import cv2
import gradio as gr
import numpy as np
from databricks.sdk import WorkspaceClient

# 環境変数からエンドポイント名を取得
MODEL_ENDPOINT_NAME = os.getenv("MODEL_ENDPOINT_NAME", "")


def convert_image_to_base64(image: np.ndarray) -> str:
    """画像をbase64エンコードされた文字列に変換する"""
    _, buffer = cv2.imencode(".jpg", image)
    return base64.b64encode(buffer).decode("utf-8")


def convert_base64_to_image(base64_string: str) -> np.ndarray:
    """base64エンコードされた文字列を画像に変換する"""
    img_data = base64.b64decode(base64_string)
    nparr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def detect_receipts(
    image: np.ndarray, endpoint_name: str, conf_threshold: float = 0.2
) -> Tuple[List, List]:
    """WorkspaceClientを使用してレシートを検出する"""
    try:
        # Databricksワークスペースクライアントの初期化
        workspace = WorkspaceClient()

        # 画像をbase64エンコード
        image_base64 = convert_image_to_base64(image)

        # リクエストデータの準備
        request_data = [{"image_data": image_base64, "conf_threshold": conf_threshold}]

        # APIエンドポイントにリクエスト
        response = workspace.serving_endpoints.query(
            name=endpoint_name, dataframe_records=request_data
        )

        # レスポンスを辞書に変換
        response_dict = response.as_dict()

        # レスポンスの確認
        if "predictions" in response_dict and len(response_dict["predictions"]) > 0:
            prediction = response_dict["predictions"][0]
            return prediction["boxes"], prediction["confidence"]
        else:
            print("APIからの応答にレシート検出結果が含まれていません")
            return [], []
    except Exception as e:
        print(f"レシート検出中にエラーが発生しました: {str(e)}")
        return [], []


def draw_bounding_boxes(
    image: np.ndarray, boxes: List, confidence: List = None
) -> np.ndarray:
    """バウンディングボックスを画像に描画する"""
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

        # 信頼度スコアの描画
        if confidence is not None and i < len(confidence):
            label_text = f"{confidence[i]:.2f}"
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


def crop_image(image: np.ndarray, box: List) -> np.ndarray:
    """バウンディングボックスに基づいて画像を切り抜く"""
    xc, yc, w, h = map(int, box)
    x1, y1, x2, y2 = map(int, [xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2])

    # 画像の範囲内に収める
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image.shape[1], x2)
    y2 = min(image.shape[0], y2)

    return image[y1:y2, x1:x2]


def create_zip_file(images):
    """複数の画像をZIPファイルにまとめる"""
    if not images:
        return None

    # 一時ファイルを作成
    temp_file = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
    temp_path = temp_file.name
    temp_file.close()

    # ZIPファイルに画像を追加
    with zipfile.ZipFile(temp_path, "w") as zip_file:
        for i, img in enumerate(images):
            # RGB -> BGR変換
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            _, buffer = cv2.imencode(".jpg", img_bgr)
            zip_file.writestr(f"receipt_{i+1}.jpg", buffer)

    return temp_path


def process_image(input_image, endpoint_name, conf_threshold):
    """画像を処理してレシート検出と結果の表示を行う"""
    if input_image is None:
        return None, None, [], None

    # 画像の前処理
    if isinstance(input_image, str):  # 画像パスの場合
        image = cv2.imread(input_image)
        # OpenCVで読み込んだ画像はBGRなのでRGBに変換
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:  # Gradioからのnumpy配列（すでにRGB）
        image_rgb = input_image
        # 検出のためにBGRに変換
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # エンドポイント名が空の場合
    if not endpoint_name:
        return image_rgb, None, [], None

    # レシート検出
    boxes, confidence = detect_receipts(image, endpoint_name, conf_threshold)

    if not boxes:
        return image_rgb, None, [], None

    # 検出結果の描画
    result_image = draw_bounding_boxes(image, boxes, confidence)
    result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

    # 切り抜き画像の作成
    cropped_images = []
    for box in boxes:
        cropped = crop_image(image, box)
        cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        cropped_images.append(cropped_rgb)

    # ZIPファイルの作成
    zip_data = create_zip_file(cropped_images)

    return (
        result_image_rgb,
        f"{len(boxes)}個のレシートが検出されました",
        cropped_images,
        zip_data,
    )


def create_app():
    """Gradioアプリケーションの作成"""
    with gr.Blocks(title="レシート検出アプリ") as app:
        gr.Markdown("# レシート検出アプリ")

        with gr.Row():
            with gr.Column(scale=3):
                # 画像アップロードエリア
                with gr.Tab("画像アップロード"):
                    input_image = gr.Image(
                        type="numpy", label="レシート画像をアップロード"
                    )
                    detect_button = gr.Button("レシートを検出", variant="primary")
                    output_image = gr.Image(type="numpy", label="検出結果")
                    result_text = gr.Textbox(label="検出結果")

                # 切り抜き画像表示エリア
                with gr.Tab("切り抜き画像"):
                    with gr.Row():
                        zip_download = gr.File(
                            label="ZIPダウンロード",
                            file_count="single",
                            type="binary",
                            interactive=False,
                        )

                    gallery = gr.Gallery(
                        label="検出されたレシート",
                        show_label=True,
                        columns=[3],
                        rows=[1],
                        object_fit="contain",
                        height="auto",
                    )

            # サイドバー（API設定）
            with gr.Column(scale=1):
                gr.Markdown("### API設定")
                endpoint_name = gr.Textbox(
                    label="エンドポイント名",
                    value=MODEL_ENDPOINT_NAME,
                    info="環境変数 MODEL_ENDPOINT_NAME から自動的に読み込まれます",
                )
                conf_threshold = gr.Slider(
                    label="信頼度閾値", minimum=0.0, maximum=1.0, value=0.1, step=0.05
                )

        # イベントハンドラの設定
        detect_result = detect_button.click(
            fn=process_image,
            inputs=[input_image, endpoint_name, conf_threshold],
            outputs=[output_image, result_text, gallery, zip_download],
        )

        # ZIPファイルが生成されたらダウンロードボタンを表示
        detect_result.then(
            fn=lambda x: gr.update(visible=x is not None),
            inputs=[zip_download],
        )

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=8000,
        root_path=os.environ.get("DATABRICKS_APP_URL"),
    )
