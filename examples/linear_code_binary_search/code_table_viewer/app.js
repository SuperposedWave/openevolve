const state = {
  data: null,
  cellsByKey: new Map(),
  selectedDetailId: null,
  viewMode: "nk",
};

const elements = {
  summary: document.getElementById("summary"),
  controls: document.getElementById("controls"),
  minN: document.getElementById("minN"),
  maxN: document.getElementById("maxN"),
  minK: document.getElementById("minK"),
  maxK: document.getElementById("maxK"),
  statusFilter: document.getElementById("statusFilter"),
  viewModeToggle: document.getElementById("viewModeToggle"),
  resetFilters: document.getElementById("resetFilters"),
  tableStatus: document.getElementById("tableStatus"),
  matrixTable: document.getElementById("matrixTable"),
  emptyDetail: document.getElementById("emptyDetail"),
  detail: document.getElementById("detail"),
  detailTitle: document.getElementById("detailTitle"),
  detailSubtitle: document.getElementById("detailSubtitle"),
  detailMetrics: document.getElementById("detailMetrics"),
  attemptList: document.getElementById("attemptList"),
  matrixSection: document.getElementById("matrixSection"),
  generatorSection: document.getElementById("generatorSection"),
  prioritySection: document.getElementById("prioritySection"),
  hRows: document.getElementById("hRows"),
  gRows: document.getElementById("gRows"),
  prioritySource: document.getElementById("prioritySource"),
  sourcePaths: document.getElementById("sourcePaths"),
  closeDetail: document.getElementById("closeDetail"),
};

function keyFor(n, k) {
  return `${n}:${k}`;
}

function clampInt(value, min, max, fallback) {
  const parsed = Number.parseInt(value, 10);
  if (Number.isNaN(parsed)) {
    return fallback;
  }
  return Math.min(Math.max(parsed, min), max);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function statusLabel(status) {
  const labels = {
    found: "found",
    failed: "searched, not found",
    upper_failed_after_found: "upper not found",
    unsearched: "unsearched",
  };
  return labels[status] || status;
}

function numberOrDash(value) {
  return value === null || value === undefined || value === "" ? "-" : value;
}

function formatTimestamp(value) {
  if (value === null || value === undefined || value === "") {
    return "-";
  }
  let date;
  if (typeof value === "number") {
    date = new Date(value < 1e12 ? value * 1000 : value);
  } else {
    date = new Date(value);
  }
  return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString();
}

function cellCoordinatesForView(viewMode, rowValue, columnValue) {
  if (viewMode === "nr") {
    const n = rowValue;
    const k = n - columnValue;
    return { n, k, valid: k >= 1 && k <= n };
  }
  return { n: rowValue, k: columnValue, valid: columnValue <= rowValue };
}

function cellMatchesFilter(cell, filters) {
  return Boolean(cell) && (filters.status === "all" || cell.status === filters.status);
}

function cellForViewCoordinate(filters, viewMode, rowValue, columnValue) {
  const coordinates = cellCoordinatesForView(viewMode, rowValue, columnValue);
  if (!coordinates.valid) {
    return null;
  }
  const { n, k } = coordinates;
  if (k < filters.minK || k > filters.maxK || n < filters.minN || n > filters.maxN) {
    return null;
  }
  return state.cellsByKey.get(keyFor(n, k)) || null;
}

function upperRunClassesForView(viewMode, cell, previousCell, nextCell) {
  if (viewMode !== "nr" || !cell) {
    return "";
  }
  const previousSame = Boolean(previousCell) && previousCell.upper === cell.upper;
  const nextSame = Boolean(nextCell) && nextCell.upper === cell.upper;
  if (!previousSame && !nextSame) {
    return "";
  }
  let classes = " upper-run";
  if (!previousSame) {
    classes += " upper-run-start";
  }
  if (!nextSame) {
    classes += " upper-run-end";
  }
  return classes;
}

function cellLabelParts(cell) {
  return String(cell.label).split("-").map((part) => ({
    text: part,
    value: Number.parseInt(part, 10),
  }));
}

function cellPartState(cell, value) {
  const bestDistance = Number.isInteger(cell.bestDistance) ? cell.bestDistance : null;
  const attemptedTargets = new Set((cell.attemptedTargets || []).map((target) => Number.parseInt(target, 10)));
  if (bestDistance !== null && value <= bestDistance) {
    return "found";
  }
  if (cell.status === "upper_failed_after_found" && value === cell.upper) {
    return "failed";
  }
  if (attemptedTargets.has(value)) {
    return "failed";
  }
  return "unsearched";
}

function cellPartClass(stateName) {
  return {
    found: "cell-value-found",
    failed: "cell-value-failed",
    unsearched: "cell-value-unsearched",
  }[stateName] || "cell-value-neutral";
}

function cellPartBackground(stateName) {
  return {
    found: "var(--found)",
    failed: "var(--failed)",
    unsearched: "#ffffff",
  }[stateName] || "#ffffff";
}

function cellBackgroundStyle(cell, matches) {
  if (!matches) {
    return "";
  }
  const parts = cellLabelParts(cell);
  if (parts.length < 2) {
    return "";
  }
  const backgrounds = parts.map((part) => cellPartBackground(cellPartState(cell, part.value)));
  const allSame = backgrounds.every((background) => background === backgrounds[0]);
  if (allSame) {
    return "";
  }
  const stops = backgrounds
    .map((background, index) => {
      const position = parts.length === 1 ? 0 : Math.round((index / (parts.length - 1)) * 100);
      return `${background} ${position}%`;
    })
    .join(", ");
  return `background: linear-gradient(90deg, ${stops});`;
}

function renderCellValue(cell, matches) {
  if (!matches) {
    return '<span class="cell-value"></span>';
  }
  const parts = cellLabelParts(cell);
  if (parts.length > 1) {
    const htmlParts = parts.map((part) => {
      const stateName = cellPartState(cell, part.value);
      return `<span class="cell-value-part ${cellPartClass(stateName)}">${escapeHtml(part.text)}</span>`;
    });
    return `<span class="cell-value cell-value-split">${htmlParts.join('<span class="cell-value-separator">-</span>')}</span>`;
  }
  return `<span class="cell-value">${escapeHtml(cell.label)}</span>`;
}

async function loadData() {
  const response = await fetch(`code_table_data.json?v=${Date.now()}`, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Failed to load code_table_data.json: ${response.status}`);
  }
  const data = await response.json();
  state.data = data;
  state.cellsByKey = new Map(data.cells.map((cell) => [keyFor(cell.n, cell.k), cell]));
}

function renderSummary() {
  const meta = state.data.meta;
  const counts = meta.countsByStatus || {};
  elements.summary.textContent =
    `${meta.totalCells.toLocaleString()} cells · ` +
    `${(counts.found || 0).toLocaleString()} found · ` +
    `${(counts.failed || 0).toLocaleString()} searched, not found · ` +
    `${(counts.upper_failed_after_found || 0).toLocaleString()} upper failed · ` +
    `${(counts.unsearched || 0).toLocaleString()} unsearched`;
}

function currentFilters() {
  let minN = clampInt(elements.minN.value, 1, 256, 1);
  let maxN = clampInt(elements.maxN.value, 1, 256, 256);
  let minK = clampInt(elements.minK.value, 1, 256, 1);
  let maxK = clampInt(elements.maxK.value, 1, 256, 256);
  if (minN > maxN) {
    [minN, maxN] = [maxN, minN];
  }
  if (minK > maxK) {
    [minK, maxK] = [maxK, minK];
  }
  return {
    minN,
    maxN,
    minK,
    maxK,
    status: elements.statusFilter.value,
  };
}

function axisValuesForView(filters) {
  if (state.viewMode === "nr") {
    const rows = [];
    for (let n = filters.minN; n <= filters.maxN; n += 1) {
      rows.push(n);
    }
    const minR = Math.max(0, filters.minN - filters.maxK);
    const maxR = Math.max(0, filters.maxN - filters.minK);
    const columns = [];
    for (let r = minR; r <= maxR; r += 1) {
      columns.push(r);
    }
    return { cornerLabel: "n/n-k", rows, columns };
  }

  const rows = [];
  for (let n = filters.minN; n <= filters.maxN; n += 1) {
    rows.push(n);
  }
  const columns = [];
  for (let k = filters.minK; k <= filters.maxK; k += 1) {
    columns.push(k);
  }
  return { cornerLabel: "n/k", rows, columns };
}

function updateViewModeControl() {
  const isRedundancyView = state.viewMode === "nr";
  elements.viewModeToggle.textContent = isRedundancyView ? "View: n/(n-k)" : "View: n/k";
  elements.viewModeToggle.setAttribute("aria-pressed", String(isRedundancyView));
}

function renderTable() {
  const filters = currentFilters();
  if (state.selectedDetailId) {
    const selectedCell = state.data.cells.find((cell) => cell.detailId === state.selectedDetailId);
    if (selectedCell && filters.status !== "all" && selectedCell.status !== filters.status) {
      state.selectedDetailId = null;
      elements.detail.hidden = true;
      elements.emptyDetail.hidden = false;
    }
  }
  const axes = axisValuesForView(filters);

  let visibleDataCells = 0;
  let matchedCells = 0;
  const header = [
    `<thead><tr><th scope="col">${axes.cornerLabel}</th>`,
    ...axes.columns.map((value) => `<th scope="col">${value}</th>`),
    "</tr></thead>",
  ].join("");

  const rows = [];
  for (let rowIndex = 0; rowIndex < axes.rows.length; rowIndex += 1) {
    const rowValue = axes.rows[rowIndex];
    const row = [`<tr><th scope="row">${rowValue}</th>`];
    for (const columnValue of axes.columns) {
      const coordinates = cellCoordinatesForView(state.viewMode, rowValue, columnValue);
      const cell = cellForViewCoordinate(filters, state.viewMode, rowValue, columnValue);
      if (!cell) {
        row.push(`<td class="blank-cell" aria-label="${coordinates.valid ? "outside selected range" : "invalid"}"></td>`);
        continue;
      }
      visibleDataCells += 1;
      const { n, k } = coordinates;
      const matches = cellMatchesFilter(cell, filters);
      if (matches) {
        matchedCells += 1;
      }
      const previousRawCell =
        rowIndex > 0
          ? cellForViewCoordinate(filters, state.viewMode, axes.rows[rowIndex - 1], columnValue)
          : null;
      const nextRawCell =
        rowIndex + 1 < axes.rows.length
          ? cellForViewCoordinate(filters, state.viewMode, axes.rows[rowIndex + 1], columnValue)
          : null;
      const previousCell = cellMatchesFilter(previousRawCell, filters) ? previousRawCell : null;
      const nextCell = cellMatchesFilter(nextRawCell, filters) ? nextRawCell : null;
      const upperRunClasses = matches
        ? upperRunClassesForView(state.viewMode, cell, previousCell, nextCell)
        : "";
      const selectedClass = cell.detailId && cell.detailId === state.selectedDetailId ? " active" : "";
      const filteredClass = matches ? "" : " filtered-cell";
      const classes = `data-cell status-${cell.status}${selectedClass}${filteredClass}${upperRunClasses}`;
      const title =
        `n=${cell.n}, k=${cell.k}, bounds=${cell.label}, status=${statusLabel(cell.status)}`;
      const backgroundStyle = cellBackgroundStyle(cell, matches);
      const styleAttribute = backgroundStyle ? ` style="${backgroundStyle}"` : "";
      if (cell.detailId && matches) {
        row.push(
          `<td class="${classes}" title="${escapeHtml(title)}"${styleAttribute}>` +
            `<button class="cell-button" type="button" data-detail-id="${cell.detailId}" ` +
            `aria-label="${escapeHtml(title)}">${renderCellValue(cell, matches)}</button>` +
          "</td>"
        );
      } else {
        row.push(
          `<td class="${classes}" title="${escapeHtml(title)}"${styleAttribute}>` +
            renderCellValue(cell, matches) +
          "</td>"
        );
      }
    }
    row.push("</tr>");
    rows.push(row.join(""));
  }

  elements.matrixTable.innerHTML = `${header}<tbody>${rows.join("")}</tbody>`;
  elements.tableStatus.textContent =
    `Showing ${matchedCells.toLocaleString()} matching cells within ` +
    `${visibleDataCells.toLocaleString()} valid cells in the selected range.`;
}

function renderMetric(name, value) {
  return `<div><dt>${escapeHtml(name)}</dt><dd>${escapeHtml(numberOrDash(value))}</dd></div>`;
}

function setActiveDetailCell(detailId, activeButton) {
  const previousActive = elements.matrixTable.querySelector(".data-cell.active");
  if (previousActive) {
    previousActive.classList.remove("active");
  }
  if (!detailId) {
    return;
  }
  const button = activeButton || Array.from(elements.matrixTable.querySelectorAll("[data-detail-id]"))
    .find((candidate) => candidate.dataset.detailId === detailId);
  const cell = button && button.closest(".data-cell");
  if (cell) {
    cell.classList.add("active");
  }
}

function renderDetails(detailId, activeButton) {
  const detail = state.data.details[detailId];
  if (!detail) {
    return;
  }
  state.selectedDetailId = detailId;
  setActiveDetailCell(detailId, activeButton);
  elements.emptyDetail.hidden = true;
  elements.detail.hidden = false;

  const status = detail.completeConstruction ? "complete construction" : "no complete construction";
  elements.detailTitle.textContent = `n=${detail.n}, k=${detail.k}`;
  elements.detailSubtitle.textContent =
    detail.trivialDistance
      ? `bounds ${detail.lower === detail.upper ? detail.lower : `${detail.lower}-${detail.upper}`} · d<=2 needs no search`
      : `bounds ${detail.lower === detail.upper ? detail.lower : `${detail.lower}-${detail.upper}`} · ${status}`;

  elements.detailMetrics.innerHTML = [
    renderMetric("best d", detail.bestDistance),
    renderMetric("target d", detail.targetDistance),
    renderMetric("lower", detail.lower),
    renderMetric("upper", detail.upper),
    renderMetric("constructed", detail.metrics.constructed_columns),
    renderMetric("score", detail.metrics.combined_score),
  ].join("");

  elements.attemptList.innerHTML = detail.attempts
    .map((attempt) => {
      const score = numberOrDash(attempt.metrics && attempt.metrics.combined_score);
      const iteration = numberOrDash(attempt.iteration);
      const generation = numberOrDash(attempt.generation);
      const source = [attempt.sourceRoot, attempt.sourceRun].filter(Boolean).join(" / ");
      const bestProgramTime = formatTimestamp(attempt.timestamp);
      const method = attempt.method ? `<br>method=${escapeHtml(attempt.method)}` : "";
      const derivedFrom = attempt.derivedFrom
        ? `<br>derived from=n=${escapeHtml(attempt.derivedFrom.n)}, k=${escapeHtml(attempt.derivedFrom.k)}, d=${escapeHtml(attempt.derivedFrom.actualDistance)}`
        : "";
      return (
        `<div class="attempt ${escapeHtml(attempt.status)}">` +
          `<strong>d=${escapeHtml(attempt.targetDistance)} · ${escapeHtml(attempt.status)}</strong>` +
          `actual=${escapeHtml(numberOrDash(attempt.actualDistance))} · score=${escapeHtml(score)} · iteration=${escapeHtml(iteration)} / generation=${escapeHtml(generation)}` +
          `${source ? `<br>source=${escapeHtml(source)}` : ""}` +
          `<br>best_program time=${escapeHtml(bestProgramTime)}` +
          method +
          derivedFrom +
        "</div>"
      );
    })
    .join("") || '<div class="notice">d<=2 lower bound is satisfied without running the search.</div>';

  if (detail.completeConstruction) {
    elements.matrixSection.hidden = false;
    elements.hRows.textContent = detail.hRows.join("\n");
    elements.generatorSection.hidden = detail.gRows.length === 0;
    elements.gRows.textContent = detail.gRows.join("\n");
    elements.prioritySection.hidden = false;
    elements.prioritySource.textContent = detail.prioritySource || "No saved priority source.";
    elements.sourcePaths.textContent = JSON.stringify(
      {
        sourceRoot: detail.sourceRoot,
        sourceRun: detail.sourceRun,
      },
      null,
      2
    );
  } else {
    elements.matrixSection.hidden = false;
    elements.hRows.textContent = detail.trivialDistance
      ? "d<=2 is satisfied without search; no saved H matrix is required."
      : "No complete saved H matrix is available for this cell.";
    elements.generatorSection.hidden = true;
    elements.gRows.textContent = "";
    elements.prioritySection.hidden = true;
    elements.prioritySource.textContent = "";
    elements.sourcePaths.textContent = detail.trivialDistance && detail.attempts.length === 0
      ? "trivial d<=2 lower-bound witness"
      : JSON.stringify(
          detail.attempts.map((attempt) => ({
            sourceRoot: attempt.sourceRoot,
            sourceRun: attempt.sourceRun,
          })),
          null,
          2
        );
  }
}

function closeDetail() {
  state.selectedDetailId = null;
  setActiveDetailCell(null);
  elements.detail.hidden = true;
  elements.emptyDetail.hidden = false;
}

function resetFilters() {
  elements.minN.value = "1";
  elements.maxN.value = "256";
  elements.minK.value = "1";
  elements.maxK.value = "256";
  elements.statusFilter.value = "all";
  renderTable();
}

function toggleViewMode() {
  state.viewMode = state.viewMode === "nk" ? "nr" : "nk";
  updateViewModeControl();
  renderTable();
}

function bindEvents() {
  elements.controls.addEventListener("input", renderTable);
  elements.controls.addEventListener("change", renderTable);
  elements.viewModeToggle.addEventListener("click", toggleViewMode);
  elements.resetFilters.addEventListener("click", resetFilters);
  elements.closeDetail.addEventListener("click", closeDetail);
  elements.matrixTable.addEventListener("click", (event) => {
    const button = event.target.closest("[data-detail-id]");
    if (!button) {
      return;
    }
    renderDetails(button.dataset.detailId, button);
  });
}

async function init() {
  try {
    await loadData();
    renderSummary();
    updateViewModeControl();
    renderTable();
    bindEvents();
  } catch (error) {
    elements.summary.textContent = error.message;
    elements.tableStatus.textContent =
      "Run this page through a local web server so it can fetch code_table_data.json.";
  }
}

init();
