"use strict";

// All rendering builds DOM nodes with textContent: model output is never parsed as HTML.

const $ = (id) => document.getElementById(id);

const state = {
  lastResponse: null,
  sourceText: "",
  activeRow: null,
};

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") node.className = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

function humanize(value) {
  if (value === null || value === undefined || value === "") return "";
  return String(value).replaceAll("_", " ").toLowerCase();
}

// ---------------------------------------------------------------- status

function setBusy(busy, message = "") {
  for (const id of ["extract", "fetch-only"]) $(id).disabled = busy;
  const status = $("status");
  status.replaceChildren();
  if (busy) status.append(el("span", { class: "spinner", "aria-hidden": "true" }));
  if (message) status.append(message);
}

function showError(message) {
  const box = $("error");
  box.textContent = message;
  box.hidden = !message;
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  let body = null;
  try {
    body = await response.json();
  } catch {
    body = null;
  }
  if (!response.ok) {
    let message = (body && (body.error || body.detail)) || `HTTP ${response.status}`;
    if (Array.isArray(message)) message = message.map((d) => d.msg || JSON.stringify(d)).join("; ");
    if (body && Array.isArray(body.details)) message += "\n" + body.details.join("\n");
    throw new Error(message);
  }
  return body;
}

// ---------------------------------------------------------------- source text

function renderSource(text, start = null, end = null) {
  const box = $("source-text");
  box.replaceChildren();
  if (!text) {
    box.append(el("div", { class: "empty" }, "No eligibility text in this record."));
    return;
  }
  if (start === null || end === null || start < 0 || end > text.length || start >= end) {
    box.append(text);
    return;
  }
  const mark = el("mark", {}, text.slice(start, end));
  box.append(text.slice(0, start), mark, text.slice(end));
  mark.scrollIntoView({ block: "center", behavior: "smooth" });
}

function selectRow(row, start, end) {
  if (state.activeRow) state.activeRow.classList.remove("active");
  state.activeRow = row;
  row.classList.add("active");
  renderSource(state.sourceText, start, end);
}

// ---------------------------------------------------------------- summary

function renderSummary(trial) {
  const card = $("summary");
  const url = `https://clinicaltrials.gov/study/${encodeURIComponent(trial.nct_id)}`;
  const enrollment = trial.enrollment && trial.enrollment.count != null
    ? `${trial.enrollment.count} (${humanize(trial.enrollment.type)})` : "";
  const ages = [trial.minimum_age || "no min", trial.maximum_age || "no max"].join(" to ");
  const fields = [
    ["Status", humanize(trial.overall_status)],
    ["Phase", (trial.phases || []).map(humanize).join(", ")],
    ["Study type", humanize(trial.study_type)],
    ["Conditions", (trial.conditions || []).join(", ")],
    ["Interventions", (trial.interventions || []).map((i) => i.name).join(", ")],
    ["Enrollment", enrollment],
    ["Sponsor", trial.lead_sponsor ? trial.lead_sponsor.name : ""],
    ["Ages", ages],
    ["Sex", humanize(trial.sex)],
    ["Healthy volunteers", trial.healthy_volunteers == null ? "" : trial.healthy_volunteers ? "yes" : "no"],
    ["Start", trial.start_date && trial.start_date.date],
    ["Locations", String(trial.num_locations ?? "")],
  ];
  card.replaceChildren(
    el("h2", {}, trial.brief_title || trial.nct_id),
    el("div", { class: "meta" },
      el("a", { href: url, target: "_blank", rel: "noopener" }, trial.nct_id),
      trial.acronym ? ` · ${trial.acronym}` : ""),
    el("dl", {}, fields.filter(([, v]) => v).map(([k, v]) => el("div", {}, el("dt", {}, k), el("dd", {}, v)))),
  );
}

function renderStats(items) {
  $("stats").replaceChildren(
    ...items.map(([label, value, warn]) =>
      el("span", { class: warn ? "stat warn" : "stat" }, `${label}: `, el("b", {}, String(value)))),
  );
}

// ---------------------------------------------------------------- default criteria

function formatTest(item) {
  if (!item.operator) return item.value != null ? String(item.value) : "";
  const unit = item.unit ? ` ${item.unit}` : "";
  if (item.operator === "between") {
    return `${item.value ?? "?"} to ${item.value_max ?? "?"}${unit}`;
  }
  if (item.operator === "present" || item.operator === "absent") {
    return item.value != null ? `${item.operator}: ${item.value}` : item.operator;
  }
  return `${item.operator} ${item.value ?? ""}${unit}`.trim();
}

function evidenceCell(evidence) {
  const notFound = evidence.status === "not_found";
  return el("div", { class: "evidence" },
    el("q", {}, evidence.source_text),
    " ",
    el("span", { class: notFound ? "badge warn" : "badge" },
      notFound ? "evidence not found" : evidence.status));
}

function criteriaTable(items) {
  if (!items.length) return el("div", { class: "empty" }, "None extracted.");
  const rows = items.map((item) => {
    const notFound = item.evidence.status === "not_found";
    const row = el("tr", {
      class: notFound ? "not-found" : null,
      tabindex: "0",
      title: notFound ? "The quoted evidence was not found in the criterion text" : "Show evidence",
    },
    el("td", {}, item.category),
    el("td", {}, el("div", {}, item.concept), evidenceCell(item.evidence)),
    el("td", { class: "mono" }, formatTest(item)),
    el("td", {}, item.time_window ? item.time_window.text : ""),
    el("td", {}, item.negated ? "yes" : ""),
    el("td", {}, item.is_biomarker ? "yes" : ""));
    const activate = () => selectRow(row, item.evidence.start, item.evidence.end);
    row.addEventListener("click", activate);
    row.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); activate(); } });
    return row;
  });
  return el("div", { class: "table-wrap" },
    el("table", {},
      el("thead", {}, el("tr", {},
        ["Category", "Concept / evidence", "Test", "Time window", "Negated", "Biomarker"].map((h) => el("th", {}, h)))),
      el("tbody", {}, rows)));
}

function renderCriteria(data) {
  const { trial, result } = data;
  $("criteria-view").hidden = false;
  $("custom-view").hidden = true;
  const inc = result.atomic.filter((a) => a.kind === "inclusion");
  const exc = result.atomic.filter((a) => a.kind === "exclusion");
  $("inclusion").replaceChildren(criteriaTable(inc));
  $("exclusion").replaceChildren(criteriaTable(exc));

  const failures = $("failures");
  failures.replaceChildren();
  if (result.failures.length) {
    failures.append(el("h2", {}, "Criteria the model could not process"),
      el("ul", {}, result.failures.map((f) => el("li", {}, `${f.criterion_id}: ${f.error}`))));
  }
  const notFound = result.atomic.filter((a) => a.evidence.status === "not_found").length;
  renderStats([
    ["Criteria", result.criteria.length],
    ["Atomic items", result.atomic.length],
    ["Evidence not found", notFound, notFound > 0],
    ["Failed criteria", result.failures.length, result.failures.length > 0],
    ["Tokens", result.usage.total_tokens],
    ["Time (s)", result.elapsed_s],
    ["Model", result.model || "n/a"],
  ]);
  state.sourceText = trial.eligibility_text;
  renderSource(state.sourceText);
}

// ---------------------------------------------------------------- custom schema

function cellValue(value) {
  if (value === null || value === undefined) return "";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function renderCustom(data) {
  const { trial, result } = data;
  $("criteria-view").hidden = true;
  $("custom-view").hidden = false;
  const checksByPath = new Map(result.evidence_checks.map((c) => [c.path, c]));

  const tables = $("custom-tables");
  tables.replaceChildren();
  for (const [key, value] of Object.entries(result.data)) {
    if (!Array.isArray(value) || !value.length || typeof value[0] !== "object" || value[0] === null) continue;
    const columns = [...new Set(value.flatMap((row) => Object.keys(row)))];
    const rows = value.map((obj, i) => {
      const checks = columns.map((c) => checksByPath.get(`$.${key}[${i}].${c}`)).filter(Boolean);
      const located = checks.find((c) => c.start != null);
      const missing = checks.some((c) => c.status === "not_found");
      const row = el("tr", { class: missing ? "not-found" : null, tabindex: "0" },
        columns.map((c) => {
          const check = checksByPath.get(`$.${key}[${i}].${c}`);
          return el("td", {}, cellValue(obj[c]),
            check && check.status === "not_found" ? [" ", el("span", { class: "badge warn" }, "not found")] : null);
        }));
      row.addEventListener("click", () => selectRow(row, located ? located.start : null, located ? located.end : null));
      return row;
    });
    tables.append(el("h2", {}, key),
      el("div", { class: "table-wrap" }, el("table", {},
        el("thead", {}, el("tr", {}, columns.map((c) => el("th", {}, c)))),
        el("tbody", {}, rows))));
  }
  $("custom-json").textContent = JSON.stringify(result.data, null, 2);

  const notFound = result.evidence_checks.filter((c) => c.status === "not_found").length;
  renderStats([
    ["Schema", result.schema_name],
    ["Evidence checks", result.evidence_checks.length],
    ["Evidence not found", notFound, notFound > 0],
    ["Output mode", result.output_mode],
    ["Repaired", result.repaired ? "yes" : "no", result.repaired],
    ["Tokens", result.usage.total_tokens],
    ["Time (s)", result.elapsed_s],
  ]);
  state.sourceText = trial.eligibility_text;
  renderSource(state.sourceText);
}

// ---------------------------------------------------------------- actions

function currentMode() {
  return document.querySelector('input[name="mode"]:checked').value;
}

function showResults(data, kind) {
  state.lastResponse = data;
  state.activeRow = null;
  $("results").hidden = false;
  $("download").disabled = false;
  renderSummary(data.trial);
  if (kind === "criteria") renderCriteria(data);
  else if (kind === "custom") renderCustom(data);
}

async function onExtract(event) {
  event.preventDefault();
  const trial = $("trial").value.trim();
  if (!trial) return;
  showError("");
  const mode = currentMode();
  setBusy(true, mode === "criteria" ? "Splitting and extracting criteria..." : "Extracting with custom schema...");
  try {
    if (mode === "criteria") {
      const data = await api("/api/criteria", { method: "POST", body: JSON.stringify({ trial }) });
      showResults(data, "criteria");
    } else {
      const body = JSON.stringify({ trial, schema_text: $("schema").value });
      const data = await api("/api/extract", { method: "POST", body });
      showResults(data, "custom");
    }
    setBusy(false, "Done.");
  } catch (err) {
    setBusy(false);
    showError(err.message);
  }
}

async function onFetchOnly() {
  const trial = $("trial").value.trim();
  if (!trial) return;
  showError("");
  setBusy(true, "Fetching record...");
  try {
    const record = await api(`/api/trial/${encodeURIComponent(trial)}`);
    state.lastResponse = { trial: record };
    $("results").hidden = false;
    $("download").disabled = false;
    renderSummary(record);
    $("criteria-view").hidden = true;
    $("custom-view").hidden = false;
    $("custom-tables").replaceChildren();
    $("custom-json").textContent = JSON.stringify(record, null, 2);
    $("stats").replaceChildren();
    state.sourceText = record.eligibility_text;
    renderSource(state.sourceText);
    setBusy(false, "Done.");
  } catch (err) {
    setBusy(false);
    showError(err.message);
  }
}

function onDownload() {
  if (!state.lastResponse) return;
  const blob = new Blob([JSON.stringify(state.lastResponse, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const name = (state.lastResponse.trial && state.lastResponse.trial.nct_id) || "result";
  const link = el("a", { href: url, download: `${name}.json` });
  document.body.append(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function onModeChange() {
  $("schema-box").hidden = currentMode() !== "custom";
}

async function loadSchemas() {
  const select = $("schema-select");
  try {
    const { examples } = await api("/api/schemas");
    const names = Object.keys(examples);
    select.replaceChildren(...names.map((n) => el("option", { value: n }, n)));
    const preferred = names.find((n) => n.includes("cognitive")) || names[0];
    if (preferred) {
      select.value = preferred;
      $("schema").value = examples[preferred];
    }
    select.addEventListener("change", () => { $("schema").value = examples[select.value] || ""; });
  } catch (err) {
    select.replaceChildren(el("option", {}, "No examples available"));
  }
}

async function loadHealth() {
  const box = $("health");
  try {
    const health = await api("/api/health");
    if (health.llm_configured) {
      box.textContent = `LLM: ${health.llm_model || "configured"}`;
      box.className = "health ok";
    } else {
      box.textContent = "LLM not configured: only 'Fetch record only' will work";
      box.className = "health warn";
    }
  } catch {
    box.textContent = "Backend unreachable";
    box.className = "health warn";
  }
}

document.addEventListener("DOMContentLoaded", () => {
  $("form").addEventListener("submit", onExtract);
  $("fetch-only").addEventListener("click", onFetchOnly);
  $("download").addEventListener("click", onDownload);
  for (const radio of document.querySelectorAll('input[name="mode"]')) radio.addEventListener("change", onModeChange);
  const params = new URLSearchParams(window.location.search);
  if (params.get("trial")) $("trial").value = params.get("trial");
  loadSchemas();
  loadHealth();
});
