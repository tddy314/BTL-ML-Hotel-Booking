const state = {
  models: [],
  selectedModel: null,
  mode: "csv",
  columns: [],
  numericColumns: [],
  sample: {},
  csvRows: [],
  manualRows: [],
  results: []
};

const els = {
  apiStatus: document.getElementById("apiStatus"),
  modelList: document.getElementById("modelList"),
  modelMetrics: document.getElementById("modelMetrics"),
  modelTemplate: document.getElementById("modelTemplate"),
  modeButtons: document.querySelectorAll(".mode-button"),
  csvMode: document.getElementById("csvMode"),
  manualMode: document.getElementById("manualMode"),
  csvFile: document.getElementById("csvFile"),
  dropzone: document.getElementById("dropzone"),
  dropzoneTitle: document.getElementById("dropzoneTitle"),
  dropzoneText: document.getElementById("dropzoneText"),
  csvInfo: document.getElementById("csvInfo"),
  downloadTemplate: document.getElementById("downloadTemplate"),
  manualRows: document.getElementById("manualRows"),
  addRow: document.getElementById("addRow"),
  rowCount: document.getElementById("rowCount"),
  predictButton: document.getElementById("predictButton"),
  validationMessage: document.getElementById("validationMessage"),
  results: document.getElementById("results"),
  resultTitle: document.getElementById("resultTitle"),
  summaryCards: document.getElementById("summaryCards"),
  resultTableHead: document.getElementById("resultTableHead"),
  resultTableBody: document.getElementById("resultTableBody"),
  downloadResults: document.getElementById("downloadResults")
};

function formatMetric(value) {
  return value == null ? "-" : Number(value).toFixed(4);
}

function formatPercent(value) {
  return value == null ? "-" : `${Number(value).toFixed(2)}%`;
}

function setStatus(text, className) {
  els.apiStatus.className = `status-pill ${className}`;
  els.apiStatus.lastChild.textContent = text;
}

async function requestJson(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "Yêu cầu không thành công.");
  }
  return data;
}

async function initialize() {
  try {
    const [modelsData, templateData] = await Promise.all([
      requestJson("/api/models"),
      requestJson("/api/template")
    ]);
    state.models = modelsData.models;
    state.columns = templateData.columns;
    state.numericColumns = templateData.numeric_columns;
    state.sample = templateData.sample;
    state.manualRows = [{ ...state.sample }];
    renderModels();
    selectModel(state.models[0].id);
    renderManualRows();
    setStatus("Mô hình sẵn sàng", "ready");
    updatePredictState();
  } catch (error) {
    setStatus("Không thể kết nối API", "error");
    els.validationMessage.textContent = error.message;
  }
}

function renderModels() {
  els.modelList.replaceChildren();
  state.models.forEach((model) => {
    const node = els.modelTemplate.content.firstElementChild.cloneNode(true);
    node.dataset.model = model.id;
    node.querySelector(".model-name").textContent = model.label;
    node.querySelector(".model-score").textContent = `F1 ${formatMetric(model.metrics.f1)} | AUC ${formatMetric(model.metrics.roc_auc)}`;
    node.addEventListener("click", () => selectModel(model.id));
    els.modelList.appendChild(node);
  });
}

function selectModel(id) {
  state.selectedModel = state.models.find((model) => model.id === id);
  document.querySelectorAll(".model-option").forEach((button) => {
    button.classList.toggle("active", button.dataset.model === id);
  });
  const model = state.selectedModel;
  const probabilityText = model.has_probability
    ? "Có xác suất dự đoán cho lớp 0 và lớp 1."
    : "LinearSVC chỉ trả decision score, chưa có xác suất đã hiệu chỉnh.";
  els.modelMetrics.innerHTML = `
    <p class="section-label">Test metrics</p>
    <div class="metric-grid">
      <div><span>Accuracy</span><strong>${formatMetric(model.metrics.accuracy)}</strong></div>
      <div><span>F1-score</span><strong>${formatMetric(model.metrics.f1)}</strong></div>
      <div><span>Recall</span><strong>${formatMetric(model.metrics.recall)}</strong></div>
      <div><span>ROC-AUC</span><strong>${formatMetric(model.metrics.roc_auc)}</strong></div>
    </div>
    <p class="probability-note">${probabilityText}</p>
  `;
  clearResults();
}

function setMode(mode) {
  state.mode = mode;
  els.modeButtons.forEach((button) => {
    button.classList.toggle("active", button.dataset.mode === mode);
  });
  els.csvMode.classList.toggle("active", mode === "csv");
  els.manualMode.classList.toggle("active", mode === "manual");
  els.validationMessage.textContent = "";
  clearResults();
  updatePredictState();
}

els.modeButtons.forEach((button) => {
  button.addEventListener("click", () => setMode(button.dataset.mode));
});

function renderManualRows() {
  els.manualRows.replaceChildren();
  state.manualRows.forEach((row, index) => {
    const detail = document.createElement("details");
    detail.className = "row-card";
    detail.open = index === state.manualRows.length - 1;
    const summary = document.createElement("summary");
    const title = document.createElement("span");
    title.textContent = `Dòng dữ liệu ${index + 1}`;
    summary.appendChild(title);
    if (state.manualRows.length > 1) {
      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "remove-row";
      remove.textContent = "Xoá";
      remove.addEventListener("click", (event) => {
        event.preventDefault();
        state.manualRows.splice(index, 1);
        renderManualRows();
      });
      summary.appendChild(remove);
    }
    detail.appendChild(summary);

    const fields = document.createElement("div");
    fields.className = "field-grid";
    state.columns.forEach((column) => {
      const wrapper = document.createElement("div");
      wrapper.className = "field";
      const label = document.createElement("label");
      label.htmlFor = `row-${index}-${column}`;
      label.textContent = column;
      const input = document.createElement("input");
      input.id = label.htmlFor;
      input.value = row[column] ?? "";
      input.type = state.numericColumns.includes(column) ? "number" : "text";
      input.step = "any";
      input.addEventListener("input", () => {
        state.manualRows[index][column] = input.value;
        updatePredictState();
      });
      wrapper.append(label, input);
      fields.appendChild(wrapper);
    });
    detail.appendChild(fields);
    els.manualRows.appendChild(detail);
  });
  els.rowCount.textContent = `${state.manualRows.length} / 10 dòng`;
  els.addRow.disabled = state.manualRows.length >= 10;
  updatePredictState();
}

els.addRow.addEventListener("click", () => {
  if (state.manualRows.length < 10) {
    state.manualRows.push({ ...state.sample });
    renderManualRows();
  }
});

function parseCsv(text) {
  const rows = [];
  let row = [];
  let value = "";
  let quoted = false;
  const source = text.replace(/^\uFEFF/, "");

  for (let i = 0; i < source.length; i += 1) {
    const char = source[i];
    const next = source[i + 1];
    if (char === "\"") {
      if (quoted && next === "\"") {
        value += "\"";
        i += 1;
      } else {
        quoted = !quoted;
      }
    } else if (char === "," && !quoted) {
      row.push(value);
      value = "";
    } else if ((char === "\n" || char === "\r") && !quoted) {
      if (char === "\r" && next === "\n") i += 1;
      row.push(value);
      if (row.some((cell) => cell.trim() !== "")) rows.push(row);
      row = [];
      value = "";
    } else {
      value += char;
    }
  }
  if (value !== "" || row.length) {
    row.push(value);
    if (row.some((cell) => cell.trim() !== "")) rows.push(row);
  }
  if (rows.length < 2) throw new Error("CSV cần có header và ít nhất một dòng dữ liệu.");

  const headers = rows[0].map((header) => header.trim());
  return rows.slice(1).map((values) => Object.fromEntries(
    headers.map((header, index) => [header, values[index] ?? ""])
  ));
}

async function handleCsvFile(file) {
  try {
    const text = await file.text();
    state.csvRows = parseCsv(text);
    els.dropzoneTitle.textContent = file.name;
    els.dropzoneText.textContent = "Tệp đã sẵn sàng để dự đoán";
    els.csvInfo.textContent = `${state.csvRows.length} dòng dữ liệu`;
    els.validationMessage.textContent = "";
    clearResults();
    updatePredictState();
  } catch (error) {
    state.csvRows = [];
    els.csvInfo.textContent = "";
    els.validationMessage.textContent = error.message;
    updatePredictState();
  }
}

els.csvFile.addEventListener("change", (event) => {
  const file = event.target.files[0];
  if (file) handleCsvFile(file);
});

["dragenter", "dragover"].forEach((eventName) => {
  els.dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    els.dropzone.classList.add("drag");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  els.dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    els.dropzone.classList.remove("drag");
  });
});

els.dropzone.addEventListener("drop", (event) => {
  const file = event.dataTransfer.files[0];
  if (file) handleCsvFile(file);
});

function toCsv(rows) {
  if (!rows.length) return "";
  const headers = Object.keys(rows[0]);
  const escape = (value) => {
    const str = value == null ? "" : String(value);
    return `"${str.replaceAll("\"", "\"\"")}"`;
  };
  return [headers.map(escape).join(","), ...rows.map((row) => headers.map((h) => escape(row[h])).join(","))].join("\n");
}

function downloadCsv(filename, rows) {
  const blob = new Blob(["\uFEFF", toCsv(rows)], { type: "text/csv;charset=utf-8" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = filename;
  link.click();
  URL.revokeObjectURL(link.href);
}

els.downloadTemplate.addEventListener("click", () => {
  downloadCsv("hotel_booking_input_template.csv", [state.sample]);
});

function updatePredictState() {
  const hasData = state.mode === "csv" ? state.csvRows.length > 0 : state.manualRows.length > 0;
  els.predictButton.disabled = !state.selectedModel || !hasData;
}

function clearResults() {
  state.results = [];
  els.results.classList.add("hidden");
}

function renderResults(response) {
  state.results = response.results;
  els.results.classList.remove("hidden");
  els.resultTitle.textContent = `${response.model_label} - ${response.results.length} dòng`;
  const canceled = response.results.filter((row) => row.predicted_class === 1).length;
  const notCanceled = response.results.length - canceled;
  const hasProbability = response.has_probability;
  const avgRisk = hasProbability
    ? response.results.reduce((sum, row) => sum + row.percent_canceled_1, 0) / response.results.length
    : null;

  const cards = [
    ["Dự đoán huỷ (1)", canceled],
    ["Không huỷ (0)", notCanceled],
    [hasProbability ? "Rủi ro huỷ trung bình" : "Xác suất", hasProbability ? formatPercent(avgRisk) : "Không hỗ trợ"]
  ];
  els.summaryCards.replaceChildren();
  cards.forEach(([label, value]) => {
    const card = document.createElement("div");
    card.className = "summary-card";
    const caption = document.createElement("span");
    caption.textContent = label;
    const number = document.createElement("strong");
    number.textContent = value;
    card.append(caption, number);
    els.summaryCards.appendChild(card);
  });

  const columns = hasProbability
    ? ["row_number", "predicted_class", "prediction", "percent_not_canceled_0", "percent_canceled_1"]
    : ["row_number", "predicted_class", "prediction", "decision_score"];
  const labels = {
    row_number: "Dòng",
    predicted_class: "Class",
    prediction: "Kết quả",
    percent_not_canceled_0: "% Không huỷ (0)",
    percent_canceled_1: "% Huỷ (1)",
    decision_score: "Decision score"
  };
  const headRow = document.createElement("tr");
  columns.forEach((column) => {
    const th = document.createElement("th");
    th.textContent = labels[column];
    headRow.appendChild(th);
  });
  els.resultTableHead.replaceChildren(headRow);
  els.resultTableBody.replaceChildren();
  response.results.forEach((result) => {
    const tr = document.createElement("tr");
    columns.forEach((column) => {
      const td = document.createElement("td");
      if (column === "percent_not_canceled_0" || column === "percent_canceled_1") {
        td.textContent = formatPercent(result[column]);
      } else if (column === "decision_score") {
        td.textContent = Number(result[column]).toFixed(4);
      } else {
        td.textContent = result[column];
      }
      if (column === "prediction") td.className = `prediction-${result.predicted_class}`;
      tr.appendChild(td);
    });
    els.resultTableBody.appendChild(tr);
  });
}

els.predictButton.addEventListener("click", async () => {
  const rows = state.mode === "csv" ? state.csvRows : state.manualRows;
  els.predictButton.disabled = true;
  els.predictButton.textContent = "Đang dự đoán...";
  els.validationMessage.textContent = "";
  try {
    const response = await requestJson("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: state.selectedModel.id,
        mode: state.mode,
        rows
      })
    });
    renderResults(response);
  } catch (error) {
    els.validationMessage.textContent = error.message;
    clearResults();
  } finally {
    els.predictButton.textContent = "Dự đoán";
    updatePredictState();
  }
});

els.downloadResults.addEventListener("click", () => {
  downloadCsv(`prediction_result_${state.selectedModel.id}.csv`, state.results);
});

initialize();
