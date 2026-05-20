/**
 * app.js — TongueCare Agent-based 工作區核心邏輯 (防呆裝甲 + 縮圖記憶版)
 */

window.addEventListener('DOMContentLoaded', () => {
    console.log('%c🚀 系統初始化開始...', 'background:#3b82f6;color:#fff;padding:4px 10px;border-radius:9999px');

    /* ── 1. DOM 快取 (視覺區塊) ───────────────────────── */
    const btnStartCamera = document.getElementById('btn-start-camera');
    const btnUpload      = document.getElementById('btn-upload');
    const fileInput      = document.getElementById('fileInput');
    const cameraStream   = document.getElementById('camera-stream');
    const previewImg     = document.getElementById('preview-img');
    const previewDefault = document.getElementById('preview-default');
    const timeline       = document.getElementById('agent-timeline');
    const cmdInput       = document.getElementById('cmd-input');

    let stream = null; 
    let currentImageBase64 = null; 

    /* ── 2. 相機與照片上傳邏輯 ───────────────────────── */
    if (btnUpload && fileInput) {
        btnUpload.addEventListener('click', () => fileInput.click());

        fileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (!file) return;
            const reader = new FileReader();
            reader.onload = ({ target }) => saveAndPreview(target.result);
            reader.readAsDataURL(file);
        });
    }

    async function startCamera() {
        try {
            stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
            cameraStream.srcObject = stream;
            previewDefault.classList.add('is-hidden');
            previewImg.classList.add('is-hidden');
            cameraStream.classList.remove('is-hidden');
            btnStartCamera.innerHTML = '<i class="fa-solid fa-camera"></i> 拍下舌象';
            btnStartCamera.onclick = takeSnapshot;
        } catch (err) {
            alert('無法存取相機，請確認您已允許瀏覽器使用相機權限。');
        }
    }

    if (btnStartCamera) btnStartCamera.onclick = startCamera;

    function takeSnapshot() {
        if (!stream) return;
        const canvas = document.createElement('canvas');
        canvas.width = cameraStream.videoWidth;
        canvas.height = cameraStream.videoHeight;
        canvas.getContext('2d').drawImage(cameraStream, 0, 0, canvas.width, canvas.height);
        
        stream.getTracks().forEach(track => track.stop());
        stream = null;
        cameraStream.classList.add('is-hidden');
        
        saveAndPreview(canvas.toDataURL('image/jpeg', 0.9));
        btnStartCamera.innerHTML = '<i class="fa-solid fa-video"></i> 重開相機';
        btnStartCamera.onclick = startCamera;
    }

    function saveAndPreview(base64Image) {
        currentImageBase64 = base64Image;
        previewImg.src = base64Image;
        previewImg.classList.remove('is-hidden');
        previewDefault.classList.add('is-hidden');
        
        appendTimelineCard('System', '📸 影像已擷取。請在下方輸入 <code>@</code> 呼叫 AI 代理進行分析。', 'system');
        document.querySelector('.agent-card:last-child')?.scrollIntoView({ behavior: 'smooth' });
    }

    /* ── 3. 增強型卡片生成器 (自動捲動 + 縮圖記憶存檔) ────────────── */
    function appendTimelineCard(sender, message, type = 'normal') {
        const initPrompt = timeline.querySelector('.fa-robot')?.parentElement;
        if (initPrompt) initPrompt.remove();

        const card = document.createElement('div');
        card.className = `agent-card agent-card--${type}`;
        
        let icon = 'fa-user-doctor';
        if (sender === 'User') icon = 'fa-user';
        else if (sender === 'System') icon = 'fa-desktop';
        else if (sender === 'Model_A' || sender === 'Model_B') icon = 'fa-network-wired';
        else if (sender === 'Gemini_TCM') icon = 'fa-leaf';

        card.innerHTML = `
            <div class="agent-card__header">
                <i class="fa-solid ${icon}"></i> <b>${sender}</b>
            </div>
            <div class="agent-card__body">${message}</div>
        `;
        timeline.appendChild(card);
        
        // 🚀 雙重保障自動捲動
        timeline.scrollTop = timeline.scrollHeight; 
        setTimeout(() => {
            card.scrollIntoView({ behavior: 'smooth', block: 'end' });
        }, 60);

        // 💾 歷史紀錄與縮圖自動存檔
        if (!message.includes('fa-spinner')) {
            localStorage.setItem('tongue_emr_history', timeline.innerHTML);
            
            // 🟢 紀錄當前照片與時間，供首頁使用
            const currentImg = window.currentImageBase64 || sessionStorage.getItem('tongue-img');
            if (currentImg) {
                localStorage.setItem('tongue_emr_image', currentImg);
            }
            localStorage.setItem('tongue_emr_time', new Date().toLocaleString());
        }
        return card;
    }

    /* ── 4. Tribute.js 指令攔截選單 ────────────────────── */
    if (cmdInput && typeof Tribute !== 'undefined') {
        const tribute = new Tribute({
            collection: [
                {
                    trigger: '@',
                    values: [
                        { key: 'Model_A', value: '分析正面舌象 (色澤/形態)' },
                        { key: 'Model_B', value: '分析舌下絡脈 (高風險特徵)' },
                        { key: 'Gemini_TCM', value: '老中醫 RAG 深度推理' }
                    ],
                    selectTemplate: function (item) { return '@' + item.original.key; }
                },
                {
                    trigger: '/',
                    values: [
                        { key: 'diagnose_full', value: '執行完整雙路徑診斷' },
                        { key: 'export_emr', value: '輸出為電子病歷格式' },
                        // 🟢 加在這裡：清空看板指令
                        { key: 'clear', value: '清空看診白板 (開啟新局)' }
                    ],
                    selectTemplate: function (item) { return '/' + item.original.key; }
                }
            ]
        });
        tribute.attach(cmdInput);

        cmdInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && cmdInput.value.trim() !== '') {
                const command = cmdInput.value.trim();
                cmdInput.value = ''; 
                triggerAgentDiagnosis(command);
            }
        });
    }

    /* ── 5. (真實連線版) 觸發診斷與 API 呼叫 ────────────── */
    //（遠端遙控模式）API 網址設定，請根據實際運行環境切換：
    //const API_URL = 'http://
    const API_URL = 'https://sneer-sureness-sandblast.ngrok-free.dev/api/analyze';

    // ⚠️ 【學長交接注意】API 網址切換指南：
    // 1. 若在「自己的電腦」獨立運行 (本機測試) -> 請用 http://127.0.0.1:8080
    // 2. 若要透過 ngrok 讓「外部手機/電腦」連線 -> 請換成 ngrok 產生的 https 網址
    //const API_URL = 'http://127.0.0.1:8080/api/analyze'; // 目前設定為：本機獨立運行模式
    //確認資料夾(/Users/.../Tongue-Diagnosis-main/.env)裡有 .env 檔案，並且裡面有放 Gemini 的 API 金鑰。並輸入% uv run python src/api_server-up.py 來喚醒大腦。
    //大腦醒了之後，他才能點開 main.html 開始玩。

    // 💾 專屬功能：匯出電子病歷
    function exportEMRToFile() {
        let text = `=========================================\n`;
        text += `    TongCare AI 舌診系統 - 電子病歷報告 (EMR)\n`;
        text += `    產出時間: ${new Date().toLocaleString()}\n`;
        text += `=========================================\n\n`;

        const cards = document.querySelectorAll('.agent-card');
        let hasContent = false;

        cards.forEach(card => {
            const senderNode = card.querySelector('.agent-card__header b');
            const bodyNode = card.querySelector('.agent-card__body');
            if (senderNode && bodyNode) {
                const sender = senderNode.innerText.trim();
                let body = bodyNode.innerText.trim();
                if (body && !body.includes('正在將資料傳送')) {
                    text += `[${sender}] :\n${body}\n`;
                    text += `-----------------------------------------\n`;
                    hasContent = true;
                }
            }
        });

        if (!hasContent) {
            alert('目前看診區尚無有效的診斷紀錄可供匯出。');
            return;
        }

        const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `舌診病歷紀錄_${new Date().toISOString().slice(0,10)}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    // 主調度器
    function triggerAgentDiagnosis(command) {
        if (command === '/export_emr') {
            appendTimelineCard('User', command, 'user');
            exportEMRToFile();
            appendTimelineCard('System', '💾 電子病歷檔案已成功生成並下載至您的裝置。', 'success');
            return;
        }

        // 🟢 新增的清空指令邏輯
        if (command === '/clear') {
            // 1. 清空畫面
            timeline.innerHTML = '';
            // 2. 清除首頁的記憶，避免回首頁又抓到舊照片
            localStorage.removeItem('tongue_emr_history');
            localStorage.removeItem('tongue_emr_image');
            localStorage.removeItem('tongue_emr_time');
            // 3. 貼上友善提示
            appendTimelineCard('System', '🧹 看診白板與暫存記憶已清空。請上傳新影像以開始下一位患者的診斷。', 'system');
            return;
        }

        const finalImageToSend = window.currentImageBase64 || sessionStorage.getItem('tongue-img') || "";

        if (!finalImageToSend && !command.includes('Gemini')) {
            appendTimelineCard('System', '⚠️ 請先上傳或拍攝舌象照片，才能進行影像分析。', 'warning');
            return;
        }

        appendTimelineCard('User', command, 'user');
        const loadingCard = appendTimelineCard('System', '<i class="fa-solid fa-spinner fa-spin"></i> 正在將資料傳送至後端運算中...', 'system');

        fetch(API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                image: finalImageToSend,
                agent_command: command
            })
        })
        .then(response => response.json()) 
        .then(data => {
            loadingCard.remove(); 
            if (data.action === 'direct_display') {
                appendTimelineCard(data.model, `結果：${data.prediction} (信心度 ${data.confidence})`, 'success');
            } else if (data.action === 'handoff') {
                appendTimelineCard(data.model, `
                    <p>預測：${data.prediction} (信心度 ${data.confidence})</p>
                    <div class="handoff-alert">⚠️ 信心度過低，自動交接給 ${data.handoff_target}</div>
                `, 'warning');
            } else {
                 appendTimelineCard('System', JSON.stringify(data), 'normal');
            }
        })
        .catch(error => {
            loadingCard.remove();
            appendTimelineCard('System', `❌ 連線失敗：${error.message}，請確認後端伺服器是否正常運作。`, 'warning');
            console.error('後端連線失敗:', error);
        });
    }

    console.log('%c🚀 TongueCare Agent 工作區已完全就緒！', 'background:#10b981;color:#fff;padding:4px 10px;border-radius:9999px');
});