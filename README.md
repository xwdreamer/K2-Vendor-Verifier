# K2 Vendor Verifier

## What's K2VV

Since the release of the Kimi K2 model, we have received numerous feedback on the precision of Kimi K2 in toolcall. Given that K2 focuses on the agentic loop, the reliability of toolcall is of utmost importance.

We have observed significant differences in the toolcall performance of various open-source solutions and vendors. When selecting a provider, users often prioritize lower latency and cost, but may inadvertently overlook more subtle yet critical differences in model accuracy.

These inconsistencies not only affect user experience but also impact K2's performance in various benchmarking results.
To mitigate these problems, we launch K2 Vendor Verifier to monitor and enhance the quality of all K2 APIs.

We hope K2VV can help ensuring that everyone can access a consistent and high-performing Kimi K2 model.


## Evaluation Results

**Test Time**: 2025-10-23

<table>
  <thead>
    <tr>
      <th rowspan="2">Model Name</th>
      <th rowspan="2">provider</th>
      <th rowspan="2">ToolCall-Trigger Similarity</th>
      <th colspan="3" style="text-align: center;">ToolCall-Schema Accuracy</th>
    </tr>
    <tr>
      <th>count_finish_reason_tool_calls</th>
      <th>count_successful_tool_call</th>
      <th>schema_accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="18">kimi-k2-0905-preview</td>
      <td><a href="https://platform.moonshot.ai/">MoonshotAI</a></td>
      <td>-</td>
      <td>1274</td>
      <td>1274</td>
      <td>100.00%</td>
    </tr>
    <tr>
      <td><a href="https://platform.moonshot.ai/">Moonshot AI Turbo</a></td>
      <td rowspan="13">>=80%</td>
      <td>1296</td>
      <td>1296</td>
      <td>100.00%</td>
    </tr>
    <tr>
      <td><a href="https://openrouter.ai/provider/deepinfra">DeepInfra</a></td>
      <td>1405</td>
      <td>1405</td>
      <td>100.00%</td>
    </tr>
    <tr>
      <td><a href="https://cloud.infini-ai.com/">Infinigence</a></td>
      <td>1249</td>
      <td>1249</td>
      <td>100.00%</td>
    </tr>
    <tr>
      <td><a href="https://openrouter.ai/provider/novita">NovitaAI</a></td>
      <td>1263</td>
      <td>1263</td>
      <td>100.00%</td>
    </tr>
    <tr>
      <td><a href="https://openrouter.ai/provider/siliconflow">SiliconFlow</a></td>
      <td>1280</td>
      <td>1276</td>
      <td>99.69%</td>
    </tr>
    <tr>
      <td><a href="https://openrouter.ai/provider/chutes">Chutes</a></td>
      <td>1225</td>
      <td>1187</td>
      <td>96.90%</td>
    </tr>
    <tr>
      <td><a href="https://github.com/vllm-project/vllm">vLLM</a></td>
      <td>1325</td>
      <td>1007</td>
      <td>76.00%</td>
    </tr>
    <tr>
      <td><a href="https://github.com/sgl-project/sglang">SGLang</a></td>
      <td>1269</td>
      <td>928</td>
      <td>73.13%</td>
    </tr>
    <tr>
      <td><a href="https://openrouter.ai/provider/ppio">PPIO</a></td>
      <td>1294</td>
      <td>945</td>
      <td>73.03%</td>
    </tr>
    <tr>
      <td><a href="https://openrouter.ai/provider/atlas-cloud">AtlasCloud</a></td>
      <td>1272</td>
      <td>925</td>
      <td>72.72%</td>
    </tr>
    <tr>
      <td><a href="https://openrouter.ai/provider/baseten">Baseten</a></td>
      <td>1363</td>
      <td>982</td>
      <td>72.05%</td>
    </tr>
    <tr>
      <td><a href="https://openrouter.ai/provider/together">Together</a></td>
      <td>1260</td>
      <td>900</td>
      <td>71.43%</td>
    </tr>
    <tr>
      <td><a href="https://www.volcengine.com/">Volc</a></td>
      <td>1344</td>
      <td>962</td>
      <td>71.58%</td>
    </tr>
    <tr>
      <td><a href="https://openrouter.ai/provider/fireworks">Fireworks</a></td>
      <td>79.68%</td>
      <td>1443</td>
      <td>1443</td>
      <td>100.00%</td>
    </tr>
    <tr>
      <td><a href="https://openrouter.ai/provider/groq">Groq</a></td>
      <td>68.21%</td>
      <td>1016</td>
      <td>1016</td>
      <td>100.00%</td>
    </tr>
    <tr>
      <td><a href="https://nebius.ai/">Nebius</a></td>
      <td>48.59%</td>
      <td>636</td>
      <td>549</td>
      <td>86.32%</td>
    </tr>
  </tbody>
</table>

##### We ran the official API multiple times to test the fluctuation of `tool_call_f1`. The lowest score was **82.71%**, and the average was **84%**. Given the inherent randomness of the model, we believe that an `tool_call_f1` score above **80%** is acceptable and can be used as a reference.


### Evaluation Metrics

#### ToolCall-Trigger Similarity
We use `tool_call_f1` to determine whether the model deployment is correct.

| Label / Metric | Formula | Meaning |
| --- | --- | --- |
| `TP` (True Positive) | — | Both model & official have `finish_reason == "tool_calls"`. |
| `FP` (False Positive) | — | Model `finish_reason == "tool_calls"` while official is `"stop"` or `"others"`. |
| `FN` (False Negative) | — | Model `finish_reason == "stop"` or `"others"` while official is `"tool_calls"`. |
| `TN` (True Negative) | — | Both model & official have `finish_reason == "stop"` or `"others"`. |
| `tool_call_precision` | `TP / (TP + FP)` | Proportion of triggered tool calls that should have been triggered. |
| `tool_call_recall` | `TP / (TP + FN)` | Proportion of tool calls that should have been triggered and were. |
| **`tool_call_f1`** | **`2*`tool_call_precision`*`tool_call_recall` / (`tool_call_precision`+`tool_call_recall`)`** | **Harmonic mean of precision and recall (primary metric for deployment check).** |



#### ToolCall-Schema Accuracy
We use `schema_accuracy` to measure the robustness of the engineering.

| Label / Metric | Formula / Condition | Description |
| --- | --- | --- |
| `count_finish_reason_tool_calls` | — | Number of responses with `finish_reason == "tool_calls"`. |
| `count_successful_tool_call` | — | Number of **tool_calls** responses that passed schema validation. |
| **`schema_accuracy`** | **`count_successful_tool_call / count_finish_reason_tool_calls`** | **Proportion of triggered tool calls whose JSON payload satisfies the schema.** |

## How we do the test

We test toolcall's response over a set of 4,000 requests. Each provider's responses are collected and compared against the official Moonshot AI API.

K2 vendors are periodically evaluated. If you are not on the list and would like to be included, feel free to contact us.

**Sample Data**: We have provided detailed sample data in samples.jsonl.

## Suggestions to Vendors

1. **Use the Correct Versions**  
Some vendors may not meet the requirements due to using incorrect versions. Here are the recommended versions:
   - [vllm v0.11.0](https://github.com/vllm-project/vllm/releases/tag/v0.11.0)
   - [sglang v0.5.3rc0](https://github.com/sgl-project/sglang/releases/tag/v0.5.3rc0)
   - [moonshotai/Kimi-K2-Instruct-0905](https://huggingface.co/moonshotai/Kimi-K2-Instruct-0905) (commit: 94a4053eb8863059dd8afc00937f054e1365abbd)

2. **Rename Tool Call IDs**  
The Kimi-K2 model expects all tool call IDs in historical messages to follow the format `functions.func_name:idx`. However, previous test cases may contain malformed tool IDs like `serach:0`*, which could mislead Kimi-K2 into generating incorrect tool call IDs, resulting in parsing failures.   
In this version, we manually add the `functions.` prefix to all previous tool calls to make Kimi-K2 happy :). We recommend that users and vendors adopt this fix in practice as well.   
This type of tool ID was generated by our official API. Before invoking the K2 model, our official API automatically renames all tool call IDs to the format `functions.func_name:idx`, so this is not an issue for us.

3. **Add Guided Encoding**  
Large language models generate text token-by-token according to probability; they have no built-in mechanism to enforce a hard JSON schema. Even with careful prompting, the model may omit fields, add extra ones, or nest them incorrectly. So please add guided encoding to ensure the correct schema.

## Verify by yourself

To run the evaluation tool with sample data, use the following command:

```bash
python tool_calls_eval.py samples.jsonl \
    --model kimi-k2-0905-preview \
    --base-url https://api.moonshot.cn/v1 \
    --api-key YOUR_API_KEY \
    --concurrency 5 \
    --output results.jsonl \
    --summary summary.json
```

- `samples.jsonl`: Path to the test set file in JSONL format
- `--model`: Model name (e.g., kimi-k2-0905-preview)
- `--base-url`: API endpoint URL
- `--api-key`: API key for authentication (or set OPENAI_API_KEY environment variable)
- `--concurrency`: Maximum number of concurrent requests (default: 5)
- `--output`: Path to save detailed results (default: results.jsonl)
- `--summary`: Path to save aggregated summary (default: summary.json)
- `--timeout`: Per-request timeout in seconds (default: 600)
- `--retries`: Number of retries on failure (default: 3)
- `--extra-body`: Extra JSON body as string to merge into each request payload (e.g., '{"temperature":0.6}')
- `--incremental`: Incremental mode to only rerun failed requests


For testing other providers via OpenRouter:

```bash
python tool_calls_eval.py samples.jsonl \
    --model moonshotai/kimi-k2-0905 \
    --base-url https://openrouter.ai/api/v1 \
    --api-key YOUR_OPENROUTER_API_KEY \
    --concurrency 5 \
    --extra-body '{"provider": {"only": ["YOUR_DESIGNATED_PROVIDER"]}}'
```
## Contact Us
**We're preparing the next benchmark round and need your input.**

If there's any **metric or test case** you care about, please drop a note in [issue](https://github.com/MoonshotAI/K2-Vendor-Verifier/issues/9)

And welcome to drop the name of any vendor you’d like to see in in [issue](https://github.com/MoonshotAI/K2-Vendor-Verifier/issues/10)

---
If you have any questions or concerns, please reach out to us at shijuanfeng@moonshot.cn.
