"""TaskHead — wraps one ONNX model + its preprocessing + label decoding."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ai.preprocessing import (
    bgr_to_rgb,
    normalise,
    resize_letterbox,
    to_chw_float,
)
from ai.types import ClassScore, HeadResult, HeadType, Normalisation


def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class TaskHead:
    session: Any                 # ort.InferenceSession or compatible mock
    name: str
    head_type: HeadType
    input_size: tuple[int, int]
    normalise: Normalisation
    class_names: list[str]
    threshold: float = 0.5
    already_probs: bool = False  # if True, skip softmax/sigmoid

    #「支援 MTL 雙輸出的程式碼補丁」
    def predict(self, image_bgr: np.ndarray) -> HeadResult:
        #raise RuntimeError("看到這行代表我改對檔案了！！") # 👈 加上這一行
        try:
            tensor = self._preprocess(image_bgr)
            input_name = self.session.get_inputs()[0].name
            
            # ⚠️ 修改點 1：拿掉逗號，因為 MTL 會有兩個輸出 (list)
            outputs = self.session.run(None, {input_name: tensor})
            return self._decode(outputs)
            
        except Exception as exc:
            return HeadResult(
                task=self.name,
                head_type=self.head_type,
                predictions=[],
                error=f"{type(exc).__name__}: {exc}",
            )
    def _preprocess(self, image_bgr: np.ndarray) -> np.ndarray:
        rgb = bgr_to_rgb(image_bgr)
        sized = resize_letterbox(rgb, self.input_size)
        floated = normalise(sized, self.normalise)
        return to_chw_float(floated)

    def _decode(self, outputs: list[np.ndarray]) -> HeadResult:
        # 🌟 修改點 2：新增 MTL 雙任務的解碼邏輯
        if getattr(self, "head_type", "") == "mtl":
            # 假設 outputs[0] 是色澤，outputs[1] 是形態
            color_logits = outputs[0][0]
            shape_logits = outputs[1][0]
            
            # 1. 顏色：維持單選 (Softmax) + 溫度縮放
            color_probs = _softmax(color_logits/ 2.0)
            c_idx = int(np.argmax(color_probs))
            color_label = self.class_names[c_idx]

            # 2. 形態 (脾虛標籤) 是關鍵！強制改用 Sigmoid + 強效溫度縮放
            print(f"DEBUG: 正在處理 MTL 形態任務，原始最高分: {np.max(shape_logits)}")
            shape_probs = _sigmoid(shape_logits / 5.0) # 👈 就是這行！
            s_idx = int(np.argmax(shape_probs))

            # c_idx = int(np.argmax(color_probs))
            # s_idx = int(np.argmax(shape_probs))

            # 從接在一起的 class_names 裡取出正確的標籤
            # 前 7 個是色澤 (index 0~6)，後 5 個是形態 (index 7~11)
            # color_label = self.class_names[c_idx]
            
            shape_label = self.class_names[7 + s_idx] #「閉著眼睛，只挑最高分的那一個，其他的全部丟掉」。

            final_shape_score = float(shape_probs[s_idx])

            print(f"DEBUG: MTL 形態最終信心度: {final_shape_score}") #完美還原單選狀態，形態最終信心度會直接顯示在網頁上，讓中醫師知道 AI 是「有信心」還是「不確定」地給出這個診斷結果
            
            # YAML 裡面的 class_names 已經改成 dict 格式：{'color': [...], 'shape': [...]}
            return HeadResult(
                task=self.name,
                head_type="mtl",
                predictions=[
                    ClassScore(label=color_label, score=float(color_probs[c_idx])),
                    ClassScore(label=shape_label, score=final_shape_score)
                ],
            )
            # 💡 [新增] 多標籤篩選邏輯
            # threshold = 0.5  # 設定及格分數 (可依臨床需求微調，例如 0.6 或 0.7)
            # shape_labels_passed = []
            # max_shape_score = 0.0  # 紀錄最高分，用來顯示在網頁的信心度上

            # for i, prob in enumerate(shape_probs):
            #     if prob >= threshold:
            #         # 只要大於門檻，就把標籤名字抓進清單
            #         shape_labels_passed.append(self.class_names[7 + i])
            #     if prob > max_shape_score:
            #         max_shape_score = float(prob) # 順便記下最高分

            # # 組合字串：如果有多個病徵，用「、」接起來；如果都沒有，就給最高分的那個
            # if not shape_labels_passed:
            #     s_idx = int(np.argmax(shape_probs))
            #     final_shape_label = self.class_names[7 + s_idx]
            # else:
            #     final_shape_label = "、".join(shape_labels_passed)

            # print(f"DEBUG: 觸發多標籤！組合後的形態為: {final_shape_label}")

            # return HeadResult(
            #     task=self.name,
            #     head_type="mtl",
            #     predictions=[
            #         ClassScore(label=color_label, score=float(color_probs[c_idx])),
            #         ClassScore(label=final_shape_label, score=max_shape_score) # 回傳組合字串 + 最高信心度
            #     ],
            # )
            
        # --- 以下保留學長原本的 single / multi 邏輯 ---
        scores = outputs[0][0] # 取出第一個輸出的 batch 0
        
        if self.head_type == "single":
            # 如果標籤只有一個，改用 sigmoid，否則用 softmax
            # if len(self.class_names) == 1:
            #     probs = _sigmoid(scores)
            # else:
            #     probs = scores if self.already_probs else _softmax(scores) # 👈 這裡！ 

            # 💡 只要任務名稱是 front 或是標籤裡有「脾虛」，我們就強制用 Sigmoid
            if "front" in self.name or "脾虛" in str(self.class_names):
                # 🌡️ 溫度縮放：把分數除以 5 或 10，強迫 AI 變得「謙虛」一點
                # 這樣 Sigmoid 算出來的數字就會落在 0.7~0.9 之間跳動了
                print(f"DEBUG: 執行溫度縮放，原始分數最大值為: {np.max(scores)}")
                probs = _sigmoid(scores / 5.0) 
            else:
                probs = scores if self.already_probs else _softmax(scores)

            idx = int(np.argmax(probs))
            
            # 🕵️‍♂️ 在終端機印出最終傳給網頁的數字，看看是不是 1.00
            print(f"DEBUG: 最終信心度為: {float(probs[idx])}")

            return HeadResult(
                task=self.name,
                head_type="single",
                predictions=[
                    ClassScore(label=self.class_names[idx], score=float(probs[idx]))
                ],
            )
        # multi
        probs = scores if self.already_probs else _sigmoid(scores)
        picks = [
            ClassScore(label=self.class_names[i], score=float(probs[i]))
            for i in range(len(self.class_names))
            if probs[i] >= self.threshold
        ]
        return HeadResult(task=self.name, head_type="multi", predictions=picks)
