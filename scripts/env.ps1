# ============================================================
# Mini-VLM 环境激活脚本（每个新终端先执行一次）
#   用法：  . .\scripts\env.ps1          <- 注意开头有一个点（dot-source）
#   作用：  固定 HF 离线变量 + 固定 python 解释器 + 固定模型路径
# ============================================================

# ---------- 1. 仓库根目录（自动推导，不写死盘符） ----------
$env:MINIVLM_ROOT = (Resolve-Path "$PSScriptRoot\..").Path

# ---------- 2. 本地离线资产目录 ----------
$env:MINIVLM_PRETRAINED = "$env:MINIVLM_ROOT\checkpoints\pretrained"
$env:MINIVLM_DATA       = "$env:MINIVLM_ROOT\data"
$env:MINIVLM_OUTPUTS    = "$env:MINIVLM_ROOT\outputs"

# ---------- 3. HuggingFace 离线三件套（核心） ----------
# HF_HOME           : 缓存根目录，放到工作区里，可整目录同步到云
# HF_HUB_OFFLINE=1  : 禁止任何 hub 网络请求；命中缓存则离线成功，否则直接报错（快速失败）
# TRANSFORMERS_OFFLINE=1 : 老版本 transformers 仍读这个变量，一起设上保险
# HF_HUB_DISABLE_TELEMETRY=1 : 关掉遥测，避免 Windows 下的无谓等待
$env:HF_HOME                   = "$env:MINIVLM_ROOT\.cache\huggingface"
$env:HF_HUB_OFFLINE            = "1"
$env:TRANSFORMERS_OFFLINE      = "1"
$env:HF_HUB_DISABLE_TELEMETRY  = "1"

# ---------- 4. 固定 python 解释器（venv 在 dl_project 下，不在 9.VLM 里） ----------
$env:MINIVLM_VENV = (Resolve-Path "$PSScriptRoot\..\..\venv").Path
$env:MINIVLM_PY   = "$env:MINIVLM_VENV\Scripts\python.exe"
if (-not (Test-Path $env:MINIVLM_PY)) {
    Write-Warning "venv python 不存在: $env:MINIVLM_PY"
}

# ---------- 5. 顺带激活 venv（可选） ----------
if (Test-Path "$env:MINIVLM_VENV\Scripts\Activate.ps1") {
    . "$env:MINIVLM_VENV\Scripts\Activate.ps1"
}

Write-Host "MINI-VLM env ready" -ForegroundColor Green
Write-Host "  root       : $env:MINIVLM_ROOT"
Write-Host "  python     : $env:MINIVLM_PY"
Write-Host "  pretrained : $env:MINIVLM_PRETRAINED"
Write-Host "  HF_HOME    : $env:HF_HOME"
Write-Host "  offline    : HF_HUB_OFFLINE=$env:HF_HUB_OFFLINE  TRANSFORMERS_OFFLINE=$env:TRANSFORMERS_OFFLINE"
Write-Host ""
Write-Host "下次直接跑：  & `$env:MINIVLM_PY scripts/check_env.py"
