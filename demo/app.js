"use strict";

const MAX_IMAGE_BYTES = 8 * 1024 * 1024;
const ALLOWED_IMAGE_TYPES = new Set(["image/jpeg", "image/png", "image/webp"]);
const TASK_NOTES = {
  existence: "请选择图片中想确认的 VOC 类别。",
  counting: "多个同类物体的计数容易出错，尤其是重叠或远处的小物体。",
  attribute: "这里的“属性”指图片中主要物体的 VOC 类别，不包含颜色等开放式属性。",
  listing: "此任务按训练涉及的 VOC 20 类理解；范围外物体不保证识别。",
  spatial: "空间位置暂不支持可靠判断；这里只展示实验性输出。",
};

const ui = {
  fileInput: document.querySelector("#image-input"),
  dropZone: document.querySelector("#drop-zone"),
  emptyUpload: document.querySelector("#empty-upload"),
  preview: document.querySelector("#image-preview"),
  imageName: document.querySelector("#image-name"),
  imageSize: document.querySelector("#image-size"),
  categoryFields: document.querySelector("#category-fields"),
  categoryField: document.querySelector("#category-field"),
  otherCategoryField: document.querySelector("#other-category-field"),
  category: document.querySelector("#category"),
  otherCategory: document.querySelector("#other-category"),
  taskNote: document.querySelector("#task-note"),
  predictButton: document.querySelector("#predict-button"),
  serviceStatus: document.querySelector("#service-status"),
  formError: document.querySelector("#form-error"),
  resultPanel: document.querySelector(".result-panel"),
  resultEmpty: document.querySelector("#result-empty"),
  resultContent: document.querySelector("#result-content"),
  resultQuestion: document.querySelector("#result-question"),
  resultAnswer: document.querySelector("#result-answer"),
  resultWarning: document.querySelector("#result-warning"),
};

const state = { imageData: null, previewUrl: null, configReady: false, busy: false, uploadSerial: 0, requestSerial: 0 };

function selectedTask() {
  return document.querySelector('input[name="task"]:checked').value;
}

function showError(message) {
  ui.formError.textContent = message;
  ui.formError.hidden = false;
}

function clearError() {
  ui.formError.textContent = "";
  ui.formError.hidden = true;
}

function setServiceStatus(message, isError = false) {
  ui.serviceStatus.textContent = message;
  ui.serviceStatus.classList.toggle("error", isError);
}

function updateButton() {
  const task = selectedTask();
  const needsTarget = task === "existence" || task === "counting" || task === "spatial";
  const categoriesValid = !needsTarget || Boolean(ui.category.value);
  const spatialValid = task !== "spatial" ||
    (Boolean(ui.otherCategory.value) && ui.category.value !== ui.otherCategory.value);
  ui.predictButton.disabled = state.busy || !state.configReady || !state.imageData || !categoriesValid || !spatialValid;
}

function updateTask() {
  state.requestSerial += 1;
  const task = selectedTask();
  const hasTarget = task === "existence" || task === "counting" || task === "spatial";
  ui.categoryFields.hidden = !hasTarget;
  ui.categoryField.hidden = !hasTarget;
  ui.otherCategoryField.hidden = task !== "spatial";
  ui.category.required = hasTarget;
  ui.otherCategory.required = task === "spatial";
  ui.taskNote.textContent = TASK_NOTES[task];
  ui.taskNote.classList.toggle("is-warning", task === "counting" || task === "spatial");
  clearError();
  clearResult();
  updateButton();
}

function fillClassSelect(select, classes) {
  select.replaceChildren();
  const placeholder = new Option("请选择 VOC 类别", "");
  placeholder.disabled = true;
  placeholder.selected = true;
  select.add(placeholder);
  for (const item of classes) {
    if (typeof item.value !== "string" || typeof item.label !== "string") continue;
    select.add(new Option(item.label, item.value));
  }
}

async function loadConfig() {
  try {
    const response = await fetch("/api/config", { cache: "no-store" });
    if (!response.ok) throw new Error(`服务返回 ${response.status}`);
    const config = await response.json();
    if (!Array.isArray(config.classes) || config.classes.length !== 20) {
      throw new Error("类别配置不完整");
    }
    fillClassSelect(ui.category, config.classes);
    fillClassSelect(ui.otherCategory, config.classes);
    state.configReady = true;
    setServiceStatus("已连接本地服务 · 请选择图片和任务");
    updateButton();
  } catch (error) {
    state.configReady = false;
    setServiceStatus(`无法连接本地服务：${error.message}。请先启动演示程序。`, true);
    updateButton();
  }
}

function readDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(new Error("图片读取失败，请重新选择。"));
    reader.readAsDataURL(file);
  });
}

function clearResult() {
  ui.resultContent.hidden = true;
  ui.resultEmpty.hidden = false;
  ui.resultQuestion.textContent = "";
  ui.resultAnswer.textContent = "";
  ui.resultWarning.textContent = "";
  ui.resultWarning.hidden = true;
}

function clearImage() {
  state.requestSerial += 1;
  state.imageData = null;
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  state.previewUrl = null;
  ui.preview.removeAttribute("src");
  ui.preview.hidden = true;
  ui.emptyUpload.hidden = false;
  ui.imageName.textContent = "尚未选择图片";
  ui.imageSize.textContent = "";
  clearResult();
  updateButton();
}

async function acceptFile(file) {
  if (!file) return;
  const serial = ++state.uploadSerial;
  clearImage();
  clearError();
  if (!ALLOWED_IMAGE_TYPES.has(file.type)) {
    showError("请选择 JPG、PNG 或 WebP 图片。");
    return;
  }
  if (file.size === 0 || file.size > MAX_IMAGE_BYTES) {
    showError("图片大小必须大于 0 且不超过 8 MB。");
    return;
  }
  try {
    const dataUrl = await readDataUrl(file);
    if (serial !== state.uploadSerial) return;
    if (typeof dataUrl !== "string" || !dataUrl.startsWith("data:image/")) {
      throw new Error("图片格式无法读取，请重新选择。");
    }
    state.previewUrl = URL.createObjectURL(file);
    ui.preview.src = state.previewUrl;
    ui.preview.hidden = false;
    ui.emptyUpload.hidden = true;
    ui.imageName.textContent = file.name;
    ui.imageSize.textContent = `${(file.size / 1024 / 1024).toFixed(2)} MB`;
    state.imageData = dataUrl;
    updateButton();
  } catch (error) {
    if (serial === state.uploadSerial) showError(error.message || "图片读取失败。");
  }
}

async function predict() {
  if (ui.predictButton.disabled) return;
  clearError();
  const task = selectedTask();
  const requestSerial = state.requestSerial;
  const payload = {
    image_data: state.imageData,
    task,
    category: ["existence", "counting", "spatial"].includes(task) ? ui.category.value : null,
    other_category: task === "spatial" ? ui.otherCategory.value : null,
  };
  state.busy = true;
  ui.predictButton.querySelector("span").textContent = "正在生成…";
  ui.resultPanel.setAttribute("aria-busy", "true");
  setServiceStatus("正在本机运行模型，首次加载可能需要一些时间…");
  updateButton();
  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    let result;
    try { result = await response.json(); }
    catch { throw new Error(`服务返回了无法读取的响应（${response.status}）。`); }
    if (requestSerial !== state.requestSerial) {
      setServiceStatus("图片或问题已更改，请重新生成回答。");
      return;
    }
    if (!response.ok) throw new Error(result.error || result.detail || `推理失败（${response.status}）。`);
    if (typeof result.question !== "string" || typeof result.answer !== "string") {
      throw new Error("服务返回的回答格式不完整。");
    }
    ui.resultQuestion.textContent = result.question;
    ui.resultAnswer.textContent = result.answer || "（模型没有输出答案）";
    const fallbackWarning = task === "spatial" ? TASK_NOTES.spatial :
      task === "counting" ? TASK_NOTES.counting : "";
    const warning = typeof result.warning === "string" && result.warning.trim()
      ? result.warning : fallbackWarning;
    ui.resultWarning.textContent = warning;
    ui.resultWarning.hidden = !warning;
    ui.resultEmpty.hidden = true;
    ui.resultContent.hidden = false;
    setServiceStatus("本地推理完成 · 请结合原图核对");
  } catch (error) {
    showError(error.message || "推理失败，请重试。");
    setServiceStatus("本地推理未完成", true);
  } finally {
    state.busy = false;
    ui.predictButton.querySelector("span").textContent = "生成回答";
    ui.resultPanel.removeAttribute("aria-busy");
    updateButton();
  }
}

ui.dropZone.addEventListener("click", () => ui.fileInput.click());
ui.dropZone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    ui.fileInput.click();
  }
});
ui.fileInput.addEventListener("change", () => {
  const file = ui.fileInput.files[0];
  ui.fileInput.value = "";
  acceptFile(file);
});
for (const type of ["dragenter", "dragover"]) {
  ui.dropZone.addEventListener(type, (event) => {
    event.preventDefault();
    ui.dropZone.classList.add("is-dragging");
  });
}
for (const type of ["dragleave", "drop"]) {
  ui.dropZone.addEventListener(type, (event) => {
    event.preventDefault();
    ui.dropZone.classList.remove("is-dragging");
  });
}
ui.dropZone.addEventListener("drop", (event) => acceptFile(event.dataTransfer.files[0]));
document.addEventListener("dragover", (event) => event.preventDefault());
document.addEventListener("drop", (event) => event.preventDefault());
document.querySelectorAll('input[name="task"]').forEach((input) => input.addEventListener("change", updateTask));
ui.category.addEventListener("change", () => { state.requestSerial += 1; clearResult(); updateButton(); });
ui.otherCategory.addEventListener("change", () => { state.requestSerial += 1; clearResult(); updateButton(); });
ui.predictButton.addEventListener("click", predict);
window.addEventListener("beforeunload", () => { if (state.previewUrl) URL.revokeObjectURL(state.previewUrl); });

updateTask();
loadConfig();
