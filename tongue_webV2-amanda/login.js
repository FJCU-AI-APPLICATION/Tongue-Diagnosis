/**
 * login.js — 登入頁邏輯
 * 負責：表單驗證、錯誤提示、忘記密碼提示
 */

/* ── DOM 快取 ─────────────────────────────────────── */
const loginErr = document.getElementById('login-err')

/* ── 工具：顯示 / 隱藏錯誤訊息 ───────────────────── */
function showErr(msg) {
    if(!loginErr) return
    loginErr.textContent = msg
    loginErr.classList.remove('is-hidden')
}

function clearErr() {
    if(!loginErr) return
    loginErr.textContent = ''
    loginErr.classList.add('is-hidden')
}

/* ── 登入表單送出 ─────────────────────────────────── */
function handleLogin(e) {
    e.preventDefault()
    clearErr()

    const email    = document.getElementById('email').value.trim()
    const password = document.getElementById('password').value

    /* 基本欄位驗證 */
    if (!email || !password) {
        showErr('請填寫帳號郵箱與密碼')
        return
    }

    /* Email 格式驗證 */
    const emailRe = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
    if (!emailRe.test(email)) {
        showErr('請輸入有效的電子郵件格式')
        return
    }

    /* ── 模擬 API 呼叫 ──────────────────────────────── */
    const submitBtn = e.target.querySelector('[type="submit"]')
    submitBtn.disabled = true
    submitBtn.textContent = '驗證中…'

    setTimeout(() => {
        /* 模擬成功切換狀態 */
        submitBtn.textContent = '✓ 登入成功'
        submitBtn.style.background = '#10b981'
        submitBtn.style.color = '#fff'

        /* 成功後跳轉回首頁或 Dashboard */
        setTimeout(() => {
            window.location.href = 'main.html'
        }, 1000)
    }, 1200)
}

/* ── 忘記密碼 ─────────────────────────────────────── */
function handleForgot() {
    const email = document.getElementById('email').value.trim()

    const overlay = document.createElement('div')
    overlay.id = 'forgot-modal'
    overlay.className = 'overlay'
    overlay.addEventListener('click', e => {
        if (e.target === overlay) overlay.remove()
    })

    overlay.innerHTML = `
        <div class="modal" onclick="event.stopImmediatePropagation()">
            <i class="fa-solid fa-envelope modal__icon"></i>
            <h3 class="modal__title">重設密碼</h3>
            <p class="modal__desc">請輸入您的帳號郵箱，我們將寄送重設連結。</p>
            <input class="field__input" id="forgot-email" type="email"
                   placeholder="帳號郵箱" value="${email}"
                   style="background:#f3f4f6;color:#111827;border-color:#e5e7eb;margin-bottom:1.5rem;">
            <button class="btn btn--primary" style="width:100%;padding:1rem"
                    onclick="submitForgot()">送出</button>
        </div>`

    document.body.appendChild(overlay)
}

function submitForgot() {
    const email = document.getElementById('forgot-email')?.value.trim()
    if (!email) return

    const modal = document.getElementById('forgot-modal')
    if (modal) modal.remove()

    /* 成功提示 */
    const notice = document.createElement('p')
    notice.className = 'login-card__err'
    notice.style.cssText = 'color:#6ee7b7;background:rgba(16,185,129,.15);border-color:rgba(16,185,129,.3);margin-top:1.5rem;'
    notice.textContent = `重設連結已寄至 ${email}`
    document.querySelector('.login-card').appendChild(notice)

    setTimeout(() => notice.remove(), 4000)
}

/* ── 初始化 ───────────────────────────────────────── */
window.addEventListener('DOMContentLoaded', () => {
    console.log('%c🔐 登入頁已載入', 'background:#1e6fa8;color:#fff;padding:4px 10px;border-radius:9999px')
})