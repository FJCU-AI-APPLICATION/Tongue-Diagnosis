/**
 * main.js — 舌診系統互動邏輯 (含 WebRTC 相機實作)
 */

/* ── DOM 快取 ─────────────────────────────────────── */
const fileInput      = document.getElementById('fileInput')
const previewImg     = document.getElementById('preview-img')
const cameraStream   = document.getElementById('camera-stream')
const previewDefault = document.getElementById('preview-default')

const actionInitial  = document.getElementById('action-initial')
const actionCamera   = document.getElementById('action-camera')
const actionReady    = document.getElementById('action-ready')

let stream = null; // 用於儲存相機影像流

/* ── 1. 檔案上傳模式 ──────────────────────────────── */
function triggerFileInput() { 
    if(fileInput) fileInput.click() 
}

function handleImageUpload(e) {
    const file = e.target.files[0]
    if (!file) return
    
    const reader = new FileReader()
    reader.onload = ({ target }) => {
        saveAndPreview(target.result)
    }
    reader.readAsDataURL(file)
}

/* ── 2. 實時相機模式 (WebRTC) ─────────────────────── */
async function startCamera() {
    try {
        // 請求相機權限（優先使用手機後置鏡頭）
        stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'environment' } 
        });
        
        cameraStream.srcObject = stream;
        
        // UI 切換
        previewDefault.classList.add('is-hidden');
        previewImg.classList.add('is-hidden');
        cameraStream.classList.remove('is-hidden');
        
        actionInitial.classList.add('is-hidden');
        actionReady.classList.add('is-hidden');
        actionCamera.classList.remove('is-hidden');

    } catch (err) {
        alert('無法存取相機，請確認您已允許瀏覽器使用相機權限。');
        console.error('Camera Error:', err);
    }
}

function takeSnapshot() {
    if (!stream) return;

    // 建立畫布來擷取當下影片影格
    const canvas = document.createElement('canvas');
    canvas.width = cameraStream.videoWidth;
    canvas.height = cameraStream.videoHeight;
    const ctx = canvas.getContext('2d');
    
    // 將影片畫面倒入 Canvas
    ctx.drawImage(cameraStream, 0, 0, canvas.width, canvas.height);

    // 轉換成 Base64 圖片字串 (品質設為 0.9)
    const base64Image = canvas.toDataURL('image/jpeg', 0.9);

    // 關閉相機並預覽
    stopCamera();
    saveAndPreview(base64Image);
}

function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
    cameraStream.classList.add('is-hidden');
}

/* ── 3. 共用：預覽與存檔邏輯 ──────────────────────── */
function saveAndPreview(imageData) {
    // 顯示圖片
    previewImg.src = imageData;
    previewImg.classList.remove('is-hidden');
    previewDefault.classList.add('is-hidden');
    
    // 切換至準備診斷狀態
    actionInitial.classList.add('is-hidden');
    actionCamera.classList.add('is-hidden');
    actionReady.classList.remove('is-hidden');
    
    // 【存檔】存入 Session Storage 供結果頁與後台使用
    sessionStorage.setItem('tongue-img', imageData);
}

function resetUpload() {
    // 清除畫面上與快取中的圖檔
    previewImg.classList.add('is-hidden');
    previewImg.src = '';
    previewDefault.classList.remove('is-hidden');
    sessionStorage.removeItem('tongue-img');
    if(fileInput) fileInput.value = '';

    // 回復初始按鈕狀態
    actionReady.classList.add('is-hidden');
    actionInitial.classList.remove('is-hidden');
}

/* ── 4. 送出後台比對 ──────────────────────────────── */
//const API_URL = 'http://127.0.0.1:8080/api/analyze'
const API_URL = 'https://sneer-sureness-sandblast.ngrok-free.dev/api/analyze'
function startDiagnosis() {
    const imageData = sessionStorage.getItem('tongue-img')
    if (!imageData) return

    const btn = document.querySelector('#action-ready .btn--primary')
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> 模型比對中...'
    btn.disabled = true

    fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        // 🟢 注意：這裡必須包含 agent_command，否則後端不知道要呼叫哪一個腦
        body: JSON.stringify({ 
            image: imageData,
            agent_command: "@Model_A" // 預設先呼叫 Model A 分析
        })
    })
    .then(response => {
        if (!response.ok) throw new Error('API 回傳錯誤')
        return response.json()
    })
    .then(data => {
        // 儲存結果後，導向我們強大的 AI 互動空間
        sessionStorage.setItem('diagnosis-result', JSON.stringify(data))
        window.location.href = 'workspace.html' 
    })
    .catch(error => {
        console.error('API 呼叫失敗：', error)
        alert('無法連線至 API，請稍後再試。')
        btn.innerHTML = '開始診斷'
        btn.disabled = false
    })
}

/* ── 初始化 ───────────────────────────────────────── */
window.addEventListener('DOMContentLoaded', () => {
    console.log('%c✅ 相機與上傳模組就緒', 'background:#10b981;color:#fff;padding:4px 10px;border-radius:9999px')
})