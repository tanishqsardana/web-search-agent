# PRISM agent integration brief

You are a coding agent working inside an application repository. Your job is to wire live tracing into PRISM so sessions, model calls, tool use, errors and latency appear in the PRISM dashboard.

Read this document fully before editing. Propose a plan, then make the smallest change that emits real traces.

## Non-negotiables

- Authenticate with the header `X-PRISMtrace-Key`. This is **not** a bearer token. `Authorization: Bearer` carries dashboard sessions and will 401 an API key.
- Do **not** invent credentials, project ids or hosts. Use exactly the values the operator supplies.
- Do **not** install an npm package. None exists. For JavaScript and TypeScript, POST to the HTTP endpoint.
- Do **not** commit secrets. `.env` stays untracked; put key **names** only in `.env.example`.
- Do **not** report success from a handshake alone. A handshake proves a credential, not an integration.
- Do **not** point production traffic at an unverified setup without explicit approval. Instrument staging first.
- Editing files is not the deliverable. A trace arriving in PRISM is.

## Required inputs

| Variable | Format | Where the operator finds it |
| --- | --- | --- |
| `PRISMTRACE_HOST` | `https://prism.blockconvey.com` | Fixed. This is the only public host. |
| `PRISMTRACE_PROJECT_ID` | UUID | Settings, Project tab |
| `PRISMTRACE_API_KEY` | `pt-sk-...` | API keys page. Shown once at creation. |

If any is missing, stop and ask. Do not guess and do not read them out of an unrelated service.

## Step 1: choose a path

Choose from repository evidence, not from what the operator says the stack is. Grep the repository. Take the first rule that fires.

| Evidence | Path |
| --- | --- |
| `langchain`, `langgraph`, `LlmAgent`, `litellm` or `agents.Runner` in Python | Python SDK |
| `anthropic.Anthropic`, `openai.OpenAI`, `google.generativeai` or `AzureOpenAI` called directly, no framework wrapper | Zero-code proxy |
| `opentelemetry`, `OTEL_EXPORTER_OTLP_ENDPOINT`, an OTel Collector config, in any language | OpenTelemetry (OTLP) |
| `package.json`, `tsconfig.json`, `.ts` / `.tsx`, Next.js, Vercel AI SDK | HTTP ingest |
| Anything else | HTTP ingest |

AWS Bedrock is none of these. It connects with AWS credentials stored in the PRISM dashboard, not from application code.

Do not run `pip install` in a repository that contains no Python.

## Step 2: instrument

### Symbols that exist

Import these from `prismtrace`. The distribution is `prismtrace-sdk`; the import package is `prismtrace`. They differ on purpose.

| Symbol | Use for |
| --- | --- |
| `PRISMtrace` | Manual client. Has `trace_llm`, `submit_trajectory` and a `trace` decorator. |
| `PRISMtraceCallbackHandler` | LangChain |
| `PRISMtraceLangGraphHandler`, `wrap_langgraph` | LangGraph |
| `PRISMtraceADKAdapter` | Google ADK |
| `install_litellm` | LiteLLM |
| `install_openai_agents` | OpenAI Agents SDK |
| `ClaudeAgentTracer` | Anthropic Messages API |
| `PRISMtraceVoiceTracer`, `install_elevenlabs_voice` | ElevenLabs voice agents |
| `PRISMtraceTracingProcessor` | OpenAI Agents, if you wire the processor yourself |
| `session`, `bind_session`, `unbind_session`, `current_session`, `resolve_session` | Ambient session identity. `with prismtrace.session("conversation-1"):` groups a run without threading an id through every call. |
| `PRISMtraceInstrumentor`, `PRISMtraceSpanExporter` | In-process OpenTelemetry, when you are already on this SDK and do not want to run an exporter |

### Symbols that do not exist

Do not write any of these. Each has been produced by a model before, and each fails on the import line.

- `PRISMtraceLangchainCallback` — the export is `PRISMtraceCallbackHandler`
- `@prismtrace/sdk`, `prismtrace-js`, or any npm package
- `monitor()`, `wrap_bedrock()`, or a `blockconvey` module
- `prism-sdk` or `prismsdk` as the pip distribution. It is `prismtrace-sdk`
- OTLP over gRPC. Only OTLP over HTTP is served, protobuf or JSON encoded, at `POST /api/otlp/v1/traces`.

### Install

```bash
pip install "prismtrace-sdk>=0.4.3"
```

Current release: 0.5.0. The floor is 0.4.3, where `prismtrace.session` and `handler.close()` arrived. Reading a framework's own conversation id (LangGraph `thread_id`, ADK `session_id`) needs 0.5.0.

### LangChain

```python
import os
import prismtrace
from prismtrace import PRISMtraceCallbackHandler

# Once, at startup. Safe to share across concurrent runs.
handler = PRISMtraceCallbackHandler(
    api_key=os.environ["PRISMTRACE_API_KEY"],
    project_id=os.environ["PRISMTRACE_PROJECT_ID"],
    host=os.environ["PRISMTRACE_HOST"],
    agent_name="my-agent",
)

with prismtrace.session("conversation-1"):
    ...  # pass callbacks=[handler] into your chain, agent or RunnableConfig

handler.close()   # at shutdown; flushes first
```

Success for LangChain is a real application invocation that flushes spans. A curl to `/api/traces` proves the credential only; it does not attach callbacks.

### LangGraph

```python
import os
import prismtrace
from prismtrace import PRISMtraceLangGraphHandler, wrap_langgraph

handler = PRISMtraceLangGraphHandler(
    api_key=os.environ["PRISMTRACE_API_KEY"],
    project_id=os.environ["PRISMTRACE_PROJECT_ID"],
    host=os.environ["PRISMTRACE_HOST"],
    agent_name="my-graph",
)
graph = wrap_langgraph(compiled_graph, handler)

with prismtrace.session("graph-run-1"):
    graph.invoke({"messages": [("user", "hello")]})

handler.close()
```

`wrap_langgraph` injects callbacks on invoke and stream, so you do not thread config through every call site. Success is one real `invoke` or `stream` that flushes spans.

**A graph that already passes `thread_id` needs no session block.** From SDK 0.5.0 the handler reads the conversation id the framework was already given, so `graph.invoke(state, config={"configurable": {"thread_id": tid}})` groups every turn of one thread onto one session. The same applies to Google ADK's `Runner.run(session_id=...)`. The precedence is fixed: an explicit `session_id=` on the handler wins, then an open `prismtrace.session(...)`, then the framework's id, then an anonymous per-handler id that groups nothing. Each trace records which of those answered as `sdk.session_origin` in its metadata, so a conversation that did not group can be diagnosed without reading the application's source.

### Grouping a conversation without threading an id

`session_id` is what turns separate traces into one conversation. If passing it through every call site is awkward, open an ambient session instead and every trace emitted inside the block inherits it.

```python
import prismtrace

with prismtrace.session("conversation-1"):
    ...  # any traced call in here is filed under conversation-1
```

Requires `prismtrace-sdk` 0.4.3 or newer. On an older install `prismtrace.session` raises `AttributeError` at the call, not at import, so pin the floor above if you use it.

Do not pass `session_id=` to the handler constructor unless the handler really is per-conversation: a constructor argument pins one value for the handler's whole life, and an app that builds one handler per agent then gets one session per agent.

### Google ADK

```python
import os
from prismtrace import PRISMtraceADKAdapter
from google.adk.agents import LlmAgent

adapter = PRISMtraceADKAdapter(
    api_key=os.environ["PRISMTRACE_API_KEY"],
    project_id=os.environ["PRISMTRACE_PROJECT_ID"],
    agent_name="my-adk-agent",
)

agent = LlmAgent(
    model="gemini-3.6-flash",
    instruction="You are a helpful assistant.",
    before_model_callback=adapter.before_model,
    after_model_callback=adapter.after_model,
    before_tool_callback=adapter.before_tool,
    after_tool_callback=adapter.after_tool,
    before_agent_callback=adapter.before_agent,
    after_agent_callback=adapter.after_agent,
)
```

ADK does not deliver errors through its callbacks. Wrap model and tool calls and forward the exception explicitly, or failed turns are missing from the trajectory:

```python
try:
    ...
except Exception as exc:
    adapter.record_model_error(exc, callback_context=ctx)
    raise
```

### LiteLLM

```python
import os, litellm
from prismtrace import install_litellm

install_litellm(
    api_key=os.environ["PRISMTRACE_API_KEY"],
    project_id=os.environ["PRISMTRACE_PROJECT_ID"],
)
```

One call at startup registers a success and failure callback. Do not wrap individual call sites; that defeats the point of LiteLLM.

### OpenAI Agents SDK

```python
import os
from agents import Agent, Runner
from prismtrace import install_openai_agents

install_openai_agents(
    api_key=os.environ["PRISMTRACE_API_KEY"],
    project_id=os.environ["PRISMTRACE_PROJECT_ID"],
)
```

### TypeScript and JavaScript

There is no package to install. Instrument the completion handler so each turn POSTs one trace.

```ts
const res = await fetch(process.env.PRISMTRACE_HOST + "/api/traces", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "X-PRISMtrace-Key": process.env.PRISMTRACE_API_KEY,
  },
  body: JSON.stringify({
    project_id: process.env.PRISMTRACE_PROJECT_ID,
    model: "gpt-4o-mini",
    input_messages: [{ role: "user", content: input }],
    output_message: output,
    latency_ms: latencyMs,
    session_id: sessionId,
  }),
});
if (!res.ok) throw new Error("PRISM ingest " + res.status + ": " + (await res.text()));
```

On serverless, await the call so the function is not frozen mid-POST.

### OpenTelemetry (OTLP)

Nothing to install and no code change. A service already instrumented with OpenTelemetry points its existing exporter at PRISM.

```bash
OTEL_EXPORTER_OTLP_ENDPOINT=https://prism.blockconvey.com/api/otlp
OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
OTEL_EXPORTER_OTLP_HEADERS=X-PRISMtrace-Key=$PRISMTRACE_API_KEY
```

Do not append `/v1/traces` yourself. The OTLP specification appends the signal path to `OTEL_EXPORTER_OTLP_ENDPOINT`, so the endpoint above resolves to `POST /api/otlp/v1/traces` on its own. Appending it by hand produces `/api/otlp/v1/traces/v1/traces`, which 404s.

Set the protocol explicitly. Both `http/protobuf` and `http/json` are served; gRPC is not, and the Java SDK defaults to it. An exporter left on gRPC batches and fails asynchronously, so the symptom is no traces and no error.

The API key alone identifies the project, so no project id is needed in the payload. If you do set one, use the resource attribute `prismtrace.project_id`, and it must belong to the same project as the key or the export is refused.

Attributes follow the OpenTelemetry GenAI semantic conventions (`gen_ai.request.model`, `gen_ai.usage.input_tokens`, `gen_ai.conversation.id` and the rest). If your instrumentation already emits them, there is nothing further to add.

**Multi-turn grouping is declared, never inferred.** Each turn arrives as its own OTel trace. Turns are stitched into one session only when spans carry `gen_ai.conversation.id`, `session.id` or `prismtrace.session_id` (span attributes first, then resource attributes). Without one there is no multi-turn trajectory to evaluate, and the export response carries `X-PRISMtrace-Sessionless-Traces`. Set the line the framework needs:

- OpenInference (LangChain, CrewAI, LlamaIndex, ADK and others): `using_session()` for `session.id`, or `session_id` / `conversation_id` / `thread_id` in the LangChain run `metadata=`. First match wins.
- LangGraph: `config["configurable"]["thread_id"]` is enough. A plain LangChain Runnable does not propagate it, so a bare chain needs the id in `metadata=` too; `metadata=` is the portable choice when you do not know which the customer has.
- openllmetry / Traceloop: set `conversation_id`, which lands in `gen_ai.conversation.id`. An id kept in `traceloop.association.properties.chat_id` needs the project setting below.
- Google ADK, agno on third-party OTel instrumentation: set `prismtrace.session_id` as a resource attribute on the exporter. On the PRISM SDK adapter, nothing: it reads ADK's session itself.
- Anything else: `prismtrace.session_id`, or the project setting below.

A project can name a non-standard key with `otlp_session_attribute` (`PATCH /api/projects/{id}`). It is read only when none of the three standard attributes is present; it never overrides a declared id and cannot help when the id was never put on the span.

Success is a real request through the instrumented service, then check 2 below. A zero-span export returns 200 and proves nothing.

### Zero-code proxy

No import and no call-site change. Swap the base URL. Guardrails then run on the call itself: a blocked request never reaches the provider, and a blocked response is withheld while the original is still recorded.

```python
import anthropic, os

client = anthropic.Anthropic(
    api_key=os.environ["ANTHROPIC_API_KEY"],
    base_url="https://prism.blockconvey.com/proxy/anthropic",
    default_headers={"X-PRISMtrace-Key": os.environ["PRISMTRACE_API_KEY"]},
)
```

Three routes exist and only three:

| Provider | Base URL |
| --- | --- |
| Anthropic | `https://prism.blockconvey.com/proxy/anthropic` |
| OpenAI | `https://prism.blockconvey.com/proxy/openai/v1` |
| Gemini | `https://prism.blockconvey.com/proxy/gemini` via `client_options={"api_endpoint": ...}` |

Azure AI Foundry uses the OpenAI route with an `x-azure-endpoint` header naming your resource.

## Step 3: verify

**This step is required.** Run both checks. Every request sends `X-PRISMtrace-Key`. If a response says "Missing bearer token" or "Invalid or expired token", you used the wrong header. Add `X-PRISMtrace-Key` and retry once. Do not invent a JWT.

### Check 1: prove the credential

```bash
curl -sS -X POST "https://prism.blockconvey.com/api/setup-doctor/handshake" \
  -H "Content-Type: application/json" \
  -H "X-PRISMtrace-Key: $PRISMTRACE_API_KEY" \
  -d '{"project_id": "'"$PRISMTRACE_PROJECT_ID"'", "send_test_trace": true}'
```

This endpoint always authenticates and its errors are specific. Read `detail` and act on it rather than retrying.

| Response | Meaning | Action |
| --- | --- | --- |
| `200 ok` | Credential valid, test trace stored | Continue to check 2 |
| `401` no credential was sent, or "No API key was sent" | Header missing or variable unset | Export it, add the header, re-run |
| `401` does not look like a PRISM API key | Wrong value pasted, often the project id | Use the `pt-sk-` value |
| `401` was revoked | Key is dead | Ask for a new key. Do not retry. |
| `401` not recognised | Truncated paste or a key never saved | Ask for a new key |
| `401` that is a PRISM API key | Sent as `Authorization: Bearer` | Use `X-PRISMtrace-Key` |
| `403` belongs to project ... | Key and `project_id` are from different projects | Use the project id in the message |
| `404` | Project does not exist | Re-check `PRISMTRACE_PROJECT_ID` |

The project is checked before the key, so a wrong `PRISMTRACE_PROJECT_ID` with a missing key answers `404`, not `401`. Fix the project id first, then read the next response.

### Check 2: confirm real traffic arrived

```bash
curl -sS "https://prism.blockconvey.com/api/setup-doctor?project_id=$PRISMTRACE_PROJECT_ID" \
  -H "X-PRISMtrace-Key: $PRISMTRACE_API_KEY"
```

In a Python repository you may run `python -m prismtrace.verify` instead. In a TypeScript repository use curl; do not install Python to verify.

On the OTLP path there is no `flush()` to call. The exporter batches on its own schedule, so allow one batch interval (5 seconds by default) between the request and this check, or force a flush through your tracer provider's shutdown. Checking immediately and reading `blocked_step: event_received` is the expected result of not waiting, not a failure.

Read `live_connected` and `blocked_step`.

| Field | Meaning |
| --- | --- |
| `live_connected: true` | A real, non-demo trace arrived. Report LIVE CONNECTED. |
| `app_connected: true` | The stricter one: the trace came from the customer's own application, not a dashboard test. Prefer it when both are present. |
| `blocked_step: event_received` | Credential works, the application sent nothing |
| `blocked_step: trace_normalized` | Traces arrived without a shared `session_id` |
| `blocked_step: analysis_ready` | Traces arrived; scoring is still catching up. Not a setup failure. |

## Step 4: report

End with these lines and nothing vaguer.

- `CREDENTIAL OK` or `CREDENTIAL FAIL — <reason>`
- `LIVE CONNECTED — PRISM is receiving live traces from <what you instrumented>.`
- or `WAITING FOR LIVE — <blocked_step> — <next action>.`

Then list files changed, remaining manual steps, and risks.

"I have edited your files" is not an acceptable final answer.

### When you cannot run the application

Proving a live trace usually means starting the application, which needs provider keys, a database and a runtime. Sandboxes often have none of these. That is not a failed setup.

If the handshake returned 200, the code is wired and credentials are in the environment and `.env.example`:

- Stop. Do not loop. Do not ask for OpenAI, Anthropic or database keys purely to verify.
- Report `CREDENTIAL OK` and `WAITING FOR LIVE — instrumented, credential proven — a human must run the application once in their own environment.`

That is an acceptable final answer. Handshake-only is never `LIVE CONNECTED`.

## API contract

Base URL `https://prism.blockconvey.com`. Auth header `X-PRISMtrace-Key` on every request.

### POST /api/traces

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `project_id` | string | yes | Project UUID |
| `model` | string | yes | Model name |
| `input_messages` | array | yes | Objects of `{role, content}` |
| `output_message` | string | yes | Agent reply |
| `latency_ms` | int | yes | Send 0 if unmeasured |
| `trace_id` | string | no | Your id. Re-sending returns the existing trace rather than duplicating. |
| `session_id` | string | no | Groups traces into one conversation. Without it nothing assembles. |
| `user_identifier` | string | no | Your end-user id |
| `agent_id` | string | no | Stable agent id. Model Inventory and agent-scoped alerts match on it. |
| `agent_name` | string | no | Display name |
| `token_count_input` | int | no | Defaults to 0 |
| `token_count_output` | int | no | Defaults to 0 |
| `metadata` | object | no | Free-form, filterable |

`session_id`, `user_identifier`, `agent_id` and `agent_name` are also read from inside `metadata` for older callers. Top level wins.

An `api_key` body field still authenticates and is deprecated; responses using it carry `X-PRISMtrace-Deprecation`. Use the header.

Returns 200 with the stored trace, including `id` and `cost_usd`.

### POST /api/spans/ingest

Used by the SDK handlers, not usually by you directly. Body is `{trace_id, project_id, spans[], session_id?, metadata?}`. Each span carries `name`, `span_type`, `start_time`, and optionally `span_id`, `parent_span_id`, `input_text`, `output_text`, `end_time`, `duration_ms`, `status`, `error_message`, `token_count_input`, `token_count_output`, `cost_usd`, `model`.

### POST /api/otlp/v1/traces

Standard OTLP/HTTP. You do not build this request; an OpenTelemetry exporter does. Point it at `/api/otlp` and the exporter appends `/v1/traces` itself.

| Aspect | Value |
| --- | --- |
| Content types | `application/x-protobuf`, `application/protobuf`, `application/json` |
| Not served | gRPC. `application/grpc` returns `415` naming the fix. |
| Auth | `X-PRISMtrace-Key`, via `OTEL_EXPORTER_OTLP_HEADERS`. The key alone identifies the project. |
| Project override | Resource attribute `prismtrace.project_id`. Checked against the key; a mismatch is refused even when other ingest paths are in soft auth mode. |
| Partial failures | Reported in the `partialSuccess` field of a `200`, never as a 4xx, because an OTLP client retries any non-2xx forever. |
| Attributes | OpenTelemetry GenAI conventions. `prismtrace.*` attributes override them. |

An empty export is legal and returns `200` with an empty body. It proves the endpoint is reachable and proves nothing about your instrumentation.

### Errors

| Status | Meaning |
| --- | --- |
| `401` | Missing or invalid credential. Body says which. |
| `403` | Valid key, wrong project, or missing scope for this call |
| `402` | `insufficient_credits` carries price and balance; `plan_limit_reached` names the limit |
| `404` | No such project |
| `429` | Rate limited. Each ingest endpoint allows 2,000 requests per minute. |

Over a plan volume ceiling the trace is still stored and the response carries `X-PRISMtrace-Plan-Warning`.

### Key scopes

| Scope | Allows |
| --- | --- |
| `ingest` | Send traces and spans. This is what an application should carry. |
| `read` | Read traces, analyses and balances |
| `operate` | Trigger actions that spend credits |

Use an `ingest`-scoped key in application code. A read-only key returns 403 on ingest by design.

## Data controls that change what you can promise

Two project settings change the outcome, and an integration report that ignores them will be wrong.

- **Content-free ingest** (Settings, Data handling). Prompt text, response text and error messages are dropped at ingest; structure, timings, token counts, cost, model and outcome are kept. A content-free trace is never sent to the LLM judge, so it gets no automatic quality scores and no Evaluators Hub scores, and is not billed for either. Deterministic guardrail rules still run. New traces only. OTLP traffic inherits the setting, and on that path it is the only control, because a stock exporter cannot ask per request.
- **Retention** can be shortened below the plan window by an org admin over the API. It cannot be lengthened self-serve.

If the operator has content-free ingest on, do not report that scoring will appear. Report that traces are arriving and that scoring is off by their own configuration.

## Account limits

Two independent systems. Plan limits are capacity; credits meter AI actions you trigger. Neither hides data already captured.

| Limit | Free | Builder |
| --- | --- | --- |
| Traces per month | 25,000 | 250,000 |
| Distinct models | 5 | 20 |
| Retention | 14 days | 90 days |
| Workspace members | 1 | 1 |
| Collaborators per project | 5 | 5 |
| Knowledge Base documents | 50 | 500 |
| Automatic analyses per day | 100 | 200 |
| Projects | unlimited | unlimited |
| Credits per 30-day cycle | 100 | 500 |

Free and Builder are the only plans. Guardrails, the Evaluators Hub and the expanded Model Inventory require Builder; everything else works on Free.

Zero credits are charged for anything you do while integrating: trace ingest, span ingest, reading traces, automatic per-trace scoring, trajectory assembly, and guardrail checks are all free. Credits are spent only on actions a human deliberately triggers in the dashboard.

Automatic scoring has a daily ceiling per workspace (100 on Free, 200 on Builder). Past it, traces still store and the deterministic checks still run; LLM scoring pauses until 00:00 UTC and the notification bell says so once. If a trace shows `skipped_quota` as its judge status, report it as the plan ceiling, not as a failure of your integration.

At a zero balance, new AI actions pause. Ingest, reads, automatic scoring, guardrails and alerts all continue. Your integration cannot be blocked by a credit balance.

## Failure modes

| Symptom | Cause | Fix |
| --- | --- | --- |
| 401 on every request | Bearer used instead of the key header | Send `X-PRISMtrace-Key` |
| 403 naming another project | Key and project id mismatched | Use the project id in the message |
| Handshake 200 but `live_connected` false | Nothing real has been sent yet | Run the application once |
| Traces arrive, no conversations | No shared `session_id` | Send one value per conversation |
| One agent appears as several | `agent_id` changes between runs | Send a stable `agent_id` |
| Import error on a PRISM symbol | The symbol does not exist | Check the symbols table above |
| `pip install` fails in a JS repo | Wrong path chosen | Use HTTP ingest |
| Scores missing on new traces | Scoring lags ingest | Wait. Not a setup failure and not billed. |
| OTLP: no traces and no error at all | Exporter defaulted to gRPC, which is not served. It batches and fails asynchronously, so nothing surfaces. | Set `OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf` |
| OTLP: `404` on export | `/v1/traces` was written into `OTEL_EXPORTER_OTLP_ENDPOINT`, so the exporter requested it twice | Set the endpoint to `.../api/otlp` and let the exporter append |
| OTLP: `415` | gRPC content type, or an encoding that is not protobuf or JSON | Read `detail`. It names the protocol to set. |
| Traces arrive with no scores, and none are coming | Content-free ingest is on for the project | Expected. Report it as the operator's setting, not a failure. |
