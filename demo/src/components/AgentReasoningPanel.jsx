/**
 * AgentReasoningPanel
 * -------------------
 * Renders the step-by-step reasoning log from a SelfImprovementAgent run.
 *
 * Props:
 *   agentRun   — AgentRunState object (from /api/demo/agent or /api/agent/:id/status)
 *   onAdvance  — callback fired when the user clicks "Next Step" (demo mode)
 */

const STEP_ICONS = {
  eval:       '📊',
  diagnose:   '🔬',
  plan:       '📋',
  dispatch:   '🚀',
  verify:     '✅',
  memory:     '💾',
  stop_check: '🛑',
};

const STATUS_CLASS = {
  pending:   'idle',
  running:   'active',
  completed: 'done',
  failed:    'error',
  skipped:   'idle',
};

const EVENT_ICONS = {
  run_started: '▶',
  iteration_started: '↻',
  step_started: '•',
  step_completed: '✓',
  iteration_completed: '▣',
  rollback: '⤺',
  run_completed: '🏁',
  run_stopped: '🛑',
};

function formatTime(isoValue) {
  if (!isoValue) return '--:--:--';
  try {
    return new Date(isoValue).toLocaleTimeString();
  } catch {
    return '--:--:--';
  }
}

function StepRow({ step }) {
  const icon  = STEP_ICONS[step.step_type] || '•';
  const cls   = STATUS_CLASS[step.status] || 'idle';
  return (
    <div className={`agent-step ${cls}`}>
      <span className="agent-step-icon">{icon}</span>
      <div className="agent-step-body">
        <div className="agent-step-label">{step.label}</div>
        {step.message && step.status !== 'pending' && (
          <div className="agent-step-msg">{step.message}</div>
        )}
      </div>
      <div className={`agent-step-badge ${cls}`}>
        {step.status === 'completed' ? '✓' : step.status === 'running' ? '…' : step.status}
      </div>
    </div>
  );
}

export default function AgentReasoningPanel({ agentRun, onAdvance, onStream, onPause, isStreaming = false }) {
  if (!agentRun) {
    return (
      <div className="card">
        <div className="section-title">Agent Reasoning</div>
        <p className="card-sub">Start Auto-Improve to see the agent reasoning step by step.</p>
      </div>
    );
  }

  const { iterations = [], status, stop_reason, reasoning_log = [], current_iteration, events = [] } = agentRun;
  const isDone = status === 'completed' || status === 'stopped';

  return (
    <div className="card agent-panel">
      {/* Header row */}
      <div className="agent-panel-header">
        <div>
          <div className="section-title">Agent Reasoning</div>
          <div className="card-sub">
            {isDone
              ? (stop_reason || 'Run complete.')
              : `Loop ${current_iteration} of ${iterations.length} — ${status}`}
          </div>
        </div>
        <div className="agent-controls">
          {!isDone && onAdvance && (
            <button className="btn-secondary" onClick={onAdvance} disabled={isStreaming}>
              Next Step ›
            </button>
          )}
          {!isDone && onStream && !isStreaming && (
            <button className="btn-secondary" onClick={onStream}>
              Start Stream
            </button>
          )}
          {!isDone && onPause && isStreaming && (
            <button className="btn-secondary" onClick={onPause}>
              Pause Stream
            </button>
          )}
        </div>
      </div>

      {events.length > 0 && (
        <div className="agent-events">
          <div className="section-title" style={{ marginBottom: 6 }}>Streaming Timeline</div>
          <div className="agent-event-list">
            {events.slice(-14).map((event) => (
              <div key={event.event_id} className={`agent-event-row ${event.event_type === 'rollback' ? 'rollback' : ''}`}>
                <span className="agent-event-icon">{EVENT_ICONS[event.event_type] || '•'}</span>
                <div className="agent-event-body">
                  <div className="agent-event-label">{event.label}</div>
                  {event.message && <div className="agent-event-msg">{event.message}</div>}
                </div>
                <div className="agent-event-meta">
                  {event.duration_ms != null && <span>{(event.duration_ms / 1000).toFixed(1)}s</span>}
                  <span>{formatTime(event.occurred_at)}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Per-loop step lists */}
      {iterations.map((loop) => (
        <div key={loop.iteration_index} className="agent-loop-block">
          <div className="agent-loop-header">
            <span className={`loop-badge ${loop.status}`}>Loop {loop.iteration_index}</span>
            <span className="agent-loop-meta">
              {loop.patch_cluster} · {loop.patch_strategy}
              {loop.eval_after != null && (
                <span className="agent-loop-uplift">
                  {' '}· {Math.round(loop.eval_before * 100)}%{' '}
                  <span style={{ color: loop.delta != null && loop.delta < 0 ? 'var(--red)' : 'var(--green)' }}>
                    → {Math.round(loop.eval_after * 100)}%
                  </span>
                  {loop.delta != null && (
                    <span style={{ color: loop.delta < 0 ? 'var(--red)' : 'var(--green)' }}>
                      {' '}({loop.delta >= 0 ? '+' : ''}{Math.round(loop.delta * 100)}pp)
                    </span>
                  )}
                  {loop.rolled_back && <span style={{ color: 'var(--red)' }}> · rolled back</span>}
                </span>
              )}
            </span>
          </div>
          {loop.rollback_reason && (
            <div className="agent-rollback-note">{loop.rollback_reason}</div>
          )}
          <div className="agent-step-list">
            {loop.steps.map((step) => (
              <StepRow key={step.step_id} step={step} />
            ))}
          </div>
        </div>
      ))}

      {/* Raw reasoning log (collapsible) */}
      {reasoning_log.length > 0 && (
        <details className="agent-log-details">
          <summary className="agent-log-summary">Full reasoning log ({reasoning_log.length} entries)</summary>
          <div className="console-list" style={{ marginTop: 8 }}>
            {reasoning_log.map((entry, i) => (
              <div key={i} className="console-item log">{entry}</div>
            ))}
          </div>
        </details>
      )}
    </div>
  );
}
