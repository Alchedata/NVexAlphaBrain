# Nvex Investor Demo Script

Use this runbook for a 10-15 minute investor demo of Nvex. The goal is not to show every feature. The goal is to make one idea unmistakable:

> Most Physical AI tools tell you what happened. Nvex tells you what to do next, executes the loop, verifies the improvement, and saves the learning.

## Demo Goal

By the end of the demo, investors should understand:

- **Problem:** Physical AI teams can train policies, but improving failed checkpoints is still slow, manual, and ad hoc.
- **Product:** Nvex is the orchestration and intelligence layer for policy improvement.
- **Proof:** The demo runs a realistic autonomous loop from **62% -> 74% -> 81% -> 79% (regression) -> rollback -> 85%** with a streamed timeline and stop condition.
- **Moat:** Every loop creates reusable platform memory: recipes, failure patterns, verification plans, and execution templates.
- **Roadmap:** Milestone 4 extends this into multi-project isolation, persistence, and customer onboarding.

## Pre-Demo Setup

Run the backend:

```bash
./.venv/bin/python -m uvicorn nvex_server.app:app --reload --port 8000
```

Run the React demo:

```bash
cd demo
npm install
npm run dev
```

Open:

```text
http://127.0.0.1:5173
```

Have this backup ready in case the local server has issues:

```bash
open demo/nvex-demo.html
```

Before the call:

- Open the demo in a clean browser window.
- Zoom to 90-100%, depending on screen size.
- Close unrelated tabs and terminals.
- Keep one terminal visible only if you want to briefly prove the backend is live.
- Start on the Project Hub page.
- Practice the transition from Failure Map to Patch Plan to Iteration Runner. That is the heart of the demo.

## Recommended Timing

| Segment | Time | Purpose |
| --- | ---: | --- |
| Opening frame | 1 min | Name the market problem |
| Project Hub | 1 min | Show Nvex as a platform, not a one-off demo |
| Project Overview | 1 min | Establish the failing checkpoint |
| Failure Map | 2 min | Show diagnosis, not dashboards |
| Patch Plan | 2 min | Show Nvex deciding the next action |
| Iteration Runner | 3 min | Show streamed autonomous execution + rollback discipline |
| Improvement Report | 2 min | Show verified uplift |
| Platform Memory | 2 min | Show compounding platform value |
| Close | 1 min | Tie to roadmap and investment thesis |

Total: 15 minutes.

## Opening Talk Track

Say:

> Physical AI is moving from model training into model improvement. Teams can get an initial robot policy working, but once it fails in the real world, the loop becomes messy: video review, benchmark logs, manual diagnosis, new data collection, retraining, and then another eval. The hard part is not just training. The hard part is knowing exactly what to fix next.

Then:

> Nvex is the orchestration layer for that loop. It takes a failing checkpoint, diagnoses why it failed, generates a targeted patch plan, dispatches the improvement run, verifies the new checkpoint, and saves what it learned so the next project starts smarter.

Optional one-liner:

> Think of Nvex as the self-improvement layer for Physical AI policies.

Optional follow-up one-liner:

> The key credibility point is not just improvement. It is safe improvement with rollback when a loop regresses.

## Page-By-Page Script

### 1. Project Hub

What to show:

- Start on the home/project hub page.
- Point to the project list and platform-level metrics.
- Do not linger. This is the map, not the story.

Say:

> This is the Nvex project hub. Each project is a policy improvement loop. The important thing is that Nvex is not just tracking experiments. It is organizing the full failure-to-fix workflow across projects.

Investor point:

> The platform gets more valuable as it sees more failures, because the memory of prior fixes becomes reusable.

Transition:

> I will walk through one concrete policy: a LIBERO Kitchen pick-and-place checkpoint that starts at 62% success.

### 2. Project Overview

What to show:

- Current checkpoint: `ckpt_v0.7`.
- Current success rate: `62%`.
- Next recommended action.

Say:

> Here Nvex has imported an evaluation artifact for a trained policy. The policy is not useless, but it is not deployable either: 62% success. This is the exact zone where teams lose time. The eval score tells you there is a problem, but not what to do next.

Investor point:

> Nvex treats eval as the beginning of the improvement loop, not the end of reporting.

Transition:

> So the next question is: what is actually failing?

### 3. Failure Map

What to show:

- Failure clusters.
- Occlusion as the top cluster.
- Recovery behavior as a secondary cluster.
- Root-cause explanation.

Say:

> Nvex compresses raw benchmark output into a failure map. In this case, failures cluster around occlusion-heavy scenes and missing recovery behavior. The policy can sometimes complete the task, but when an object is partially obstructed or the first grasp fails, it does not recover reliably.

Then:

> This is the first key distinction: Nvex is not just showing charts. It is converting raw eval results into an actionable diagnosis.

Investor point:

> The product wedge is post-training intelligence: diagnosis, prioritization, and targeted improvement.

Transition:

> Once Nvex knows why the policy fails, it can generate the patch plan.

### 4. Patch Plan

What to show:

- Data recipe: occlusion-heavy patch episodes.
- Recovery traces / teleop corrections.
- Training strategy: continual learning or fine-tune update.
- Verification plan and expected uplift.

Say:

> This is the patch plan generated from the failure map. Nvex recommends targeted data, not random more-data collection. For this checkpoint, it proposes 120 occlusion-heavy episodes and 40 recovery correction trajectories, then a continual-learning update and a verification pass.

Then:

> The key is that this plan is structured. It can be reviewed by a human, dispatched to an execution backend, and reused later if it works.

Investor point:

> This is where Nvex becomes more than analytics. It turns diagnosis into an executable improvement plan.

Transition:

> Now we move from plan to execution.

### 5. Iteration Runner

What to show:

- Streaming timeline events (`run_started`, `step_started`, `step_completed`, `rollback`, `run_stopped`).
- Auto-stream controls (start/pause) and variable step durations.
- Four-loop arc with one intentional regression and rollback.
- Logs/artifacts and backend-driven execution path if asked.

Say:

> The iteration runner is the operating heart of Nvex. It streams each reasoning and execution event in order, so you can see what the agent is doing and why, not just a final score.

Then:

> Notice this run is realistic, not perfectly monotonic. The third loop regresses from 81% to 79%. Nvex triggers rollback automatically, reverts to the prior checkpoint baseline, and launches a safer corrective loop.

Be precise:

> For investor demos, this run is seeded and replayable so timing is stable. The event model, schemas, dispatch interface, polling, rollback signaling, and report generation are implemented. For customer POCs, this is where we connect their checkpoint and run the same loop asynchronously.

Investor point:

> Nvex is designed to sit above execution frameworks. It owns decisioning, governance, and loop control. Execution backends can vary.

Transition:

> After the run finishes, Nvex does not just say "job complete." It verifies whether the policy actually improved.

### 6. Improvement Report

What to show:

- Loop-by-loop: `62 -> 74 -> 81 -> 79 (rollback) -> 85`.
- Final uplift: `+23pp` versus the starting checkpoint.
- Regression loop rendered in red and marked as rolled back.
- Cluster reduction / recovery improvement and generated assets.
- Generated assets.

Say:

> Here is the result. Nvex climbs from 62% to 85%, but importantly it does not hide failure. It surfaces a regression, applies rollback, and recovers safely. That behavior is critical for customer trust.

Then:

> This matters because Physical AI improvement needs to be auditable and controlled. You want to know not only that the score improved, but where it regressed, why rollback was triggered, and which recipe is now safe to reuse.

Investor point:

> Nvex makes improvement measurable and repeatable. That is what turns a services-like process into software.

Transition:

> The most important screen for the company thesis is the last one: Platform Memory.

### 7. Platform Memory

What to show:

- Recipes.
- Failure ontology.
- Pipeline templates.
- Compounding chart or memory assets.

Say:

> Every loop deposits reusable assets into Platform Memory: a patch recipe, a failure pattern, a verification setup, and execution metadata. The next time Nvex sees a similar occlusion or recovery failure, it does not start from scratch.

Then:

> This is the compounding loop. More projects produce more failures. More failures produce more recipes. More recipes make future improvement faster and more reliable.

Add:

> The failed loop is also memory. Nvex stores anti-patterns so teams avoid repeating known bad interventions.

Investor point:

> The moat is not one benchmark result. The moat is the accumulated memory of how to fix Physical AI failures across tasks, environments, embodiments, and customers.

## Closing Script

Say:

> The takeaway is simple: Physical AI teams need a self-improvement layer. Training frameworks are necessary, but they do not decide what to fix next. Nvex starts from failure, diagnoses the gap, generates the plan, runs the iteration, verifies the result, and saves the learning.

Then:

> Today, the demo shows the full autonomous loop on a seeded LIBERO Kitchen case: 62% to 85% with a visible regression and rollback. The next milestone extends this into customer-grade multi-project operation: project isolation, persistent memory, and onboarding for bring-your-own checkpoints and eval artifacts.

Final line:

> We are building the operating layer for Physical AI systems that learn from every failure.

## Short Version: 5-Minute Script

Use this if time is tight.

1. **Open:** "Physical AI teams can train policies, but improving failed checkpoints is still manual. Nvex closes that loop."
2. **Overview:** "This checkpoint starts at 62% success. The eval score tells us it failed, but Nvex tells us why."
3. **Failure Map:** "Failures cluster around occlusion and missing recovery behavior."
4. **Patch Plan:** "Nvex generates targeted data, training, and verification steps instead of asking the team to guess."
5. **Runner:** "Nvex dispatches the improvement job and tracks artifacts."
6. **Report:** "The run reaches 85%, including one regression that Nvex rolls back automatically."
7. **Memory:** "The fix becomes reusable platform memory, so every loop makes the system smarter."
8. **Close:** "This is the self-improvement layer for Physical AI."

## Investor Q&A Prep

### "Is this just an MLOps dashboard?"

Answer:

> No. MLOps tracks runs and artifacts. Nvex decides what to do next. The core product surface is diagnosis, patch planning, orchestration, verification, and reusable memory.

### "Is this just a wrapper around AlphaBrain?"

Answer:

> No. AlphaBrain is one execution backend bundled in this repo. Nvex owns the intelligence layer above execution: failure maps, patch plans, iteration control, improvement reports, and platform memory. Over time, Nvex can dispatch to multiple training and eval backends.

### "What is real today?"

Answer:

> The current demo includes a React product surface and FastAPI backend path with schema contracts, eval import, patch-plan generation, job dispatch, polling, multi-iteration agent state, streamed timeline events, rollback signaling, and report generation. The investor run is replayable for stability; customer POCs connect real checkpoints and run asynchronously.

### "Why will this compound?"

Answer:

> Every loop creates structured assets: failure patterns, recipes, verification specs, and execution templates. Those assets reduce the time and uncertainty of future loops. That is especially important because robotics failures repeat across tasks and environments.

### "Who is the first customer?"

Answer:

> Robotics and Physical AI teams with an initial trained policy and a painful post-training loop: benchmark failures, field failures, or sim-to-real regressions where they need targeted improvement rather than another undirected training run.

### "What is the wedge?"

Answer:

> Start with post-eval diagnosis and patch planning for VLA manipulation policies. The product expands from "tell me why this checkpoint failed" to "run the improvement loop for me."

### "What is the autonomous agent?"

Answer:

> The agent runs eval, diagnoses failures, selects interventions, dispatches training, verifies the checkpoint, saves memory, emits timeline events, and decides whether to continue, rollback, or stop. It is targeted incremental improvement, not retraining from scratch.

### "Why now?"

Answer:

> Physical AI is moving from demos to deployment. As policies enter more varied environments, the bottleneck shifts from initial model training to continuous improvement after failure. Teams need infrastructure for that loop.

## Phrases To Use

- "Eval is the beginning of the improvement loop, not the end of reporting."
- "Nvex turns failure into an executable patch plan."
- "We are not replacing training frameworks; we are orchestrating them."
- "The product compounds because each fix becomes memory."
- "The demo is stable and replayable; the architecture is designed for live customer POCs."
- "We show real control discipline: if a loop regresses, Nvex rolls back and recovers."
- "The moat is the growing library of failure patterns and successful interventions."

## Phrases To Avoid

- Avoid implying this seeded investor flow is a live multi-hour training run.
- Avoid promising real-time training during an investor meeting.
- Avoid positioning Nvex as only a benchmark dashboard.
- Avoid making AlphaBrain the center of the story. Mention it as the bundled execution layer only if asked or during the runner section.
- Avoid claiming the 62% to 85% case proves general deployment readiness. It proves loop quality, control, and product thesis.

## Backup Plan

If the backend fails:

1. Open `demo/nvex-demo.html`.
2. Say:

   > I am switching to the static walkthrough so we can keep the story moving. It shows the same product flow and seeded improvement case.

3. Continue from the same page sequence.

If the UI is slow:

1. Skip the Project Overview.
2. Go directly: Failure Map -> Patch Plan -> Improvement Report -> Platform Memory.
3. Use the 5-minute script.

If someone asks for implementation detail:

> The backend is FastAPI. The key objects are eval runs, failure diagnoses, patch plans, iteration jobs, improvement reports, and reusable memory assets. The dispatch layer is intentionally separated so Nvex can orchestrate different execution backends.

If they ask about control and safety:

> The agent emits explicit run events and applies rollback when a loop regresses beyond tolerance. That is the foundation for customer-facing governance and audit trails.

## One-Slide Summary

Use this as the verbal summary if you only get one minute:

> Nvex is the self-improvement layer for Physical AI. It starts with a failing checkpoint, diagnoses the failure modes, generates a targeted patch plan, dispatches the improvement run, verifies the new checkpoint, and saves the recipe to platform memory. In the demo, a LIBERO Kitchen policy improves from 62% to 74%. The long-term thesis is that every failure makes the platform smarter, creating a compounding library of recipes for robot policy improvement.

Replace the metric callout if using the current Milestone 4 demo flow:

> In the current demo, a LIBERO Kitchen policy progresses from 62% to 85%, including a visible regression and automatic rollback before final convergence.
