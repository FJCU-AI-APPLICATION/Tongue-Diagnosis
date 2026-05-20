import base64
from io import BytesIO
import io
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import onnxruntime as ort
from PIL import Image

# 🟢 [新增] 1. 引入讀取環境變數 (.env) 與 Gemini API 的套件
import os
from dotenv import load_dotenv, find_dotenv
import google.generativeai as genai

# 🌟 精準引入 TaskHead 大腦
from ai.task_head import TaskHead
from ai.types import HeadType, Normalisation


# 🟢 [新增] 2. 伺服器啟動時，讀取金鑰並準備好 Gemini 模型
# 讓 Python 像雷達一樣，自動去整棟大樓找出 .env 檔案在哪裡！
load_dotenv(find_dotenv())
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')
    print("✨ Gemini API 連線準備就緒！")
else:
    gemini_model = None
    print("⚠️ 警告：找不到 GEMINI_API_KEY，請確認 .env 檔案")


app = FastAPI(title="TongueCare AI Agent Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🌟 初始化 ONNX 雙頭模型大腦 (完全沒動)
print("🧠 正在初始化 ONNX 雙頭模型大腦...")
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    onnx_model_path = os.path.join(current_dir, "ai", "models", "best_resnet50_front_mtl.onnx")

    session = ort.InferenceSession(onnx_model_path)

    my_brain = TaskHead(
        session=session,
        name="front",
        head_type="mtl",
        input_size=(224, 224),
        normalise=Normalisation(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
        class_names=[
            "淡紅", "紅", "淡", "絳", "青紫", "暗", "微紅",
            "脾虛濕標籤", "瘦薄", "偏斜", "瘀血絲", "無異常",
        ],
    )
    print("✅ ONNX 大腦初始化成功！隨時可以看診！")
except Exception as e:
    print(f"❌ 大腦初始化失敗，請檢查 ONNX 模型路徑是否正確。錯誤：{e}")


# 🟢 修正 1：讓 image 變成可有可無 (Optional)
class AgentRequest(BaseModel):
    image: Optional[str] = ""
    agent_command: str  

@app.post("/api/analyze")
async def analyze_tongue(request: AgentRequest):
    base64_image_str = request.image
    command = request.agent_command

    print(f"📥 收到 Agent 指令: {command}")

    # 🟢 修正 2：準備預設變數，避免沒傳圖片時找不到這些值而崩潰
    real_color = "未知"
    real_shape = "未知"
    real_color_score = 0.0
    real_shape_score = 0.0

    try:
        # 🟢 修正 3：加入防護網！有圖片才解碼，沒圖片直接跳過這裡
        if base64_image_str and len(base64_image_str) > 50:
            if "," in base64_image_str:
                base64_image_str = base64_image_str.split(",")[1]

            image_bytes = base64.b64decode(base64_image_str)
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            open_cv_image = np.array(pil_image)
            image_bgr = open_cv_image[:, :, ::-1].copy()

            # 真正呼叫 ONNX 進行預測
            result = my_brain.predict(image_bgr)

            real_color = result.predictions[0].label
            real_color_score = float(result.predictions[0].score) 

            real_shape = result.predictions[1].label
            real_shape_score = float(result.predictions[1].score)

            print(f"🎯 AI 預測 -> 顏色: {real_color}({real_color_score:.2f}), 形態: {real_shape}({real_shape_score:.2f})")
        else:
            print("⚠️ 未收到圖片資料，使用預設值直接執行指令邏輯")


        # 🌟 L2 漸進式揭露邏輯
        if "@Model_A" in command or "/diagnose_full" in command:
            CONFIDENCE_THRESHOLD = 0.70
            if real_color_score >= CONFIDENCE_THRESHOLD and real_shape_score >= CONFIDENCE_THRESHOLD:
                return {
                    "model": "Model_A",
                    "action": "direct_display",
                    "prediction": f"色澤: {real_color}, 形態: {real_shape}",
                    "confidence": round(min(real_color_score, real_shape_score), 2)
                }
            else:
                return {
                    "model": "Model_A",
                    "action": "handoff",
                    "prediction": f"色澤: {real_color}, 形態: {real_shape}",
                    "confidence": round(min(real_color_score, real_shape_score), 2),
                    "handoff_target": "@Gemini_TCM"
                }
                
        elif "@Model_B" in command:
            return {
                "model": "Model_B",
                "action": "direct_display",
                "prediction": "需使用舌下絡脈專用模型進行判讀",
                "confidence": 1.0
            }
            
        elif "@Gemini_TCM" in command:
            # 🟢 [修改] 3. 拔除假資料，組合提示詞 (Prompt) 進行真實 Gemini 呼叫
            if not gemini_model:
                return {
                    "model": "System",
                    "action": "direct_display",
                    "prediction": "無法呼叫老中醫：後端缺少 API 金鑰。",
                    "confidence": 0.0
                }
                
            # 將 ONNX 算出來的真實特徵，餵給老中醫
            prompt = f"你是一位專業且親切的中醫師。患者目前的舌診影像分析結果為：色澤為「{real_color}」，形態特徵呈現「{real_shape}」。請根據這兩個特徵，用大約 60 到 80 字給出中醫證型推論以及簡單的日常飲食調理建議。"
            
            print("🤖 正在向 Gemini 請求深度推理中...")
            try:
                # 正式呼叫 Google 伺服器
                response = gemini_model.generate_content(prompt)
                tcm_advice = response.text
                
                return {
                    "model": "Gemini_TCM",
                    "action": "direct_display",
                    "prediction": f"【老中醫深度推理】<br>{tcm_advice}",
                    "confidence": 0.95
                }
            except Exception as api_err:
                print(f"❌ Gemini API 錯誤: {api_err}")
                return {
                    "model": "Gemini_TCM",
                    "action": "handoff",
                    "prediction": "呼叫 Google 伺服器超時或失敗，請稍後再試。",
                    "confidence": 0.0,
                    "handoff_target": "System"
                }

        else:
            return {
                "model": "System",
                "action": "direct_display",
                "prediction": f"大腦尚未學習如何處理此指令：{command}",
                "confidence": 1.0
            }

    except Exception as e:
        print(f"❌ 診斷過程中發生錯誤: {e}")
        return {
            "model": "System",
            "action": "handoff",
            "prediction": "影像解析錯誤",
            "confidence": 0.0,
            "handoff_target": "System Admin"
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)