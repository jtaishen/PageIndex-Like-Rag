import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";

const OBSERVED_TOOLS = new Set([
  "kb_tree_search",
  "kb_compare",
  "kb_generate_review",
  "kb_check_review_citations",
  "kb_eval_search",
  "kb_eval_review",
  "kb_eval_memory",
  "kb_eval_facts",
  "kb_create_eval_suite",
  "kb_run_benchmark",
  "kb_analyze_failures",
  "kb_generate_case_study",
  "kb_get_table_content",
  "kb_get_table_summaries",
  "kb_put_feedback",
  "kb_build_eval_set_from_feedback",
  "kb_eval_dashboard",
  "kb_tune_search",
  "kb_apply_search_profile",
  "kb_get_search_profile",
]);

const SENSITIVE_KEYS = new Set([
  "answer",
  "comment",
  "content",
  "excerpt",
  "evidence",
  "review_draft",
  "section_drafts",
  "snippet",
  "rows",
  "cells",
  "table_content",
  "table_summaries",
  "text",
  "tree_search_trace",
]);

export const KbObserverPlugin = async ({ directory }) => {
  const stateDir = path.join(directory, ".kb_state", "opencode_observer");
  const latestPath = path.join(stateDir, "latest.json");

  async function readLatest() {
    try {
      return JSON.parse(await readFile(latestPath, "utf8"));
    } catch {
      return { schema: "kb_observer_state.v1", events: [] };
    }
  }

  async function writeLatest(state) {
    await mkdir(stateDir, { recursive: true });
    await writeFile(latestPath, JSON.stringify(state, null, 2), "utf8");
  }

  return {
    async "tool.execute.after"(input, output) {
      const toolName = normalizeToolName(input.tool);
      if (!OBSERVED_TOOLS.has(toolName)) return;
      const parsed = parseToolOutput(output);
      const event = summarizeToolEvent(toolName, input, parsed);
      if (!event) return;
      const state = await readLatest();
      const events = Array.isArray(state.events) ? state.events : [];
      events.push(event);
      const next = {
        schema: "kb_observer_state.v1",
        updated_at: new Date().toISOString(),
        session_id: input.sessionID,
        events: events.slice(-12),
      };
      await writeLatest(next);
      output.metadata = {
        ...(output.metadata || {}),
        kb_observer: {
          recorded: true,
          tool: toolName,
          warning_count: event.warning_count,
          task_id: event.task_id || "",
        },
      };
    },

    async "experimental.session.compacting"(_input, output) {
      const state = await readLatest();
      const events = Array.isArray(state.events) ? state.events.slice(-5) : [];
      if (!events.length) return;
      output.context.push(formatCompactionContext(events));
    },
  };
};

export default KbObserverPlugin;

function normalizeToolName(tool) {
  const raw = String(tool || "");
  for (const name of OBSERVED_TOOLS) {
    if (raw === name || raw.endsWith(`.${name}`) || raw.endsWith(`_${name}`) || raw.endsWith(`-${name}`)) {
      return name;
    }
  }
  return raw;
}

function parseToolOutput(output) {
  const candidates = [output?.metadata?.result, output?.output];
  for (const candidate of candidates) {
    if (!candidate) continue;
    if (typeof candidate === "object") return candidate;
    if (typeof candidate === "string") {
      const trimmed = candidate.trim();
      if (!trimmed.startsWith("{") && !trimmed.startsWith("[")) continue;
      try {
        return JSON.parse(trimmed);
      } catch {
        continue;
      }
    }
  }
  return {};
}

function summarizeToolEvent(tool, input, payload) {
  const safe = sanitize(payload || {});
  const taskId = stringValue(safe.task_id || safe?.review_outline?.task_id || safe?.comparison_matrix?.task_id);
  const warnings = arrayValue(safe.warnings || safe?.review_outline?.warnings || safe?.comparison_matrix?.warnings);
  const coverage = safe.evidence_coverage || safe?.review_outline?.evidence_coverage || safe?.comparison_matrix?.evidence_coverage || {};
  return {
    schema: "kb_observer_event.v1",
    recorded_at: new Date().toISOString(),
    session_id: input.sessionID,
    call_id: input.callID,
    tool,
    task_id: taskId,
    status: stringValue(safe.status),
    query: stringValue(safe.query || safe.topic || input?.args?.query || input?.args?.topic),
    doc_id: stringValue(safe.doc_id || input?.args?.doc_id),
    warning_count: warnings.length,
    warnings: warnings.slice(0, 8),
    metrics: summarizeMetrics(tool, payload || {}, coverage),
    profile: profileSummary(tool, safe),
    feedback_hint: feedbackHint(tool, safe, warnings, coverage),
  };
}

function summarizeMetrics(tool, safe, coverage) {
  if (tool === "kb_tree_search") {
    return {
      result_count: numberValue(safe.results?.length),
      evidence_count: numberValue(safe.evidence?.length),
      latency_ms: numberValue(safe.latency_ms),
    };
  }
  if (tool === "kb_compare" || tool === "kb_generate_review") {
    return {
      selected_paper_count: numberValue(safe?.selected_papers?.paper_count),
      source_doc_count: numberValue(coverage.source_doc_count),
      total_evidence_count: numberValue(coverage.total_evidence_count || coverage.cells_with_evidence),
    };
  }
  if (tool === "kb_check_review_citations" || tool === "kb_eval_review") {
    return {
      citation_coverage_score: numberValue(safe.citation_coverage_score || safe?.review_report?.citation_coverage_score),
      missing_ref_count: numberValue(safe.missing_ref_count),
      unsupported_paragraph_count: numberValue(safe.unsupported_paragraph_count),
    };
  }
  if (tool === "kb_eval_facts") {
    return {
      total_fact_count: numberValue(safe.total_fact_count),
      table_backed_fact_count: numberValue(safe.table_backed_fact_count),
      low_confidence_count: numberValue(safe.low_confidence_count),
      duplicate_group_count: numberValue(safe.duplicate_group_count),
    };
  }
  if (tool === "kb_create_eval_suite") {
    return {
      query_count: numberValue(safe.query_count),
      source_count: numberValue(safe.sources?.length),
      warning_count: numberValue(safe.warnings?.length),
    };
  }
  if (tool === "kb_run_benchmark") {
    return {
      query_count: numberValue(safe.query_count),
      mode_count: numberValue(safe.compare_modes?.length),
      best_mode_by_score: stringValue(safe.best_mode_by_score),
      best_mode_by_node_recall: stringValue(safe.best_mode_by_node_recall),
      warning_count: numberValue(safe.warnings?.length),
    };
  }
  if (tool === "kb_analyze_failures") {
    return {
      failure_count: numberValue(safe.failure_count),
      next_action_count: numberValue(safe.next_actions?.length),
    };
  }
  if (tool === "kb_generate_case_study") {
    return {
      mode_count: numberValue(safe.compare_modes?.length),
      evidence_count: numberValue(safe?.evidence_summary?.count),
      fact_match_count: numberValue(safe?.fact_matches?.count),
      warning_count: numberValue(safe.warnings?.length),
    };
  }
  if (tool === "kb_get_table_content" || tool === "kb_get_table_summaries") {
    return {
      table_count: numberValue(safe.count),
      table_warning_count: numberValue(safe.table_warning_count),
    };
  }
  return {
    query_count: numberValue(safe.query_count),
    fallback_count: numberValue(safe.fallback_count),
    suspected_pollution_count: numberValue(safe.suspected_pollution_count),
    feedback_count: numberValue(safe.feedback_count || safe?.feedback_summary?.feedback_count),
    low_rating_count: numberValue(safe.low_rating_count || safe?.feedback_summary?.low_rating_count),
    mode_count: numberValue(safe.compare_modes?.length || safe.mode_rankings?.length),
  };
}

function profileSummary(tool, safe) {
  if (tool === "kb_tune_search") {
    return {
      default_mode: stringValue(safe.default_mode),
      profile_name: stringValue(safe?.saved_profile?.name),
      path: stringValue(safe?.saved_profile?.path || safe.path),
    };
  }
  if (tool === "kb_apply_search_profile" || tool === "kb_get_search_profile") {
    return {
      default_mode: stringValue(safe.default_mode || safe?.profile?.default_mode),
      profile_name: stringValue(safe.name || safe?.profile?.name || safe?.active?.name),
      path: stringValue(safe.path || safe?.profile?.path || safe?.active?.path),
    };
  }
  return {};
}

function feedbackHint(tool, safe, warnings, coverage) {
  const evidenceCount = numberValue(safe.evidence?.length || safe.evidence_count || coverage.total_evidence_count);
  const fallbackCount = numberValue(safe.fallback_count);
  const unsupported = numberValue(safe.unsupported_paragraph_count);
  if (tool === "kb_put_feedback" || tool === "kb_build_eval_set_from_feedback") return "";
  if (warnings.length || fallbackCount || unsupported || evidenceCount === 0) {
    return "Use kb_put_feedback for representative failures, then kb_build_eval_set_from_feedback and kb_eval_search to compare modes.";
  }
  return "";
}

function sanitize(value) {
  if (Array.isArray(value)) return value.map(sanitize).filter((item) => item !== undefined).slice(0, 40);
  if (!value || typeof value !== "object") return value;
  const result = {};
  for (const [key, item] of Object.entries(value)) {
    if (SENSITIVE_KEYS.has(key)) continue;
    result[key] = sanitize(item);
  }
  return result;
}

function formatCompactionContext(events) {
  const lines = [
    "KB observer context: recent knowledge-base tool outcomes. Do not treat this as paper evidence; use it only for task recovery and quality warnings.",
  ];
  for (const event of events) {
    const parts = [
      `tool=${event.tool}`,
      event.task_id ? `task_id=${event.task_id}` : "",
      event.status ? `status=${event.status}` : "",
      event.warning_count ? `warnings=${event.warnings.join(";")}` : "",
      event.profile?.profile_name ? `profile=${event.profile.profile_name}:${event.profile.default_mode}` : "",
      event.feedback_hint ? `feedback_hint=${event.feedback_hint}` : "",
    ].filter(Boolean);
    lines.push(`- ${parts.join(" ")}`);
  }
  return lines.join("\n");
}

function stringValue(value) {
  return value == null ? "" : String(value).slice(0, 240);
}

function arrayValue(value) {
  return Array.isArray(value) ? value.map(String) : [];
}

function numberValue(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : 0;
}
