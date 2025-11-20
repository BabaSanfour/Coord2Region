# AI Provider Configuration

Coord2Region discovers and activates AI providers through environment
variables. The recommended workflow is to generate a private
`config/coord2region-config.yaml` using:

```
python scripts/configure_coord2region.py
```

The helper asks for the credentials you wish to enable, stores them in the
YAML file (which is ignored by Git), and exports them to the environment when
Coord2Region starts. You can still manage variables manually via your shell,
secret manager, or a legacy `.env` file if desired.

| Provider | Environment variable(s) | Notes | Docs |
| --- | --- | --- | --- |
| OpenAI | `OPENAI_API_KEY` | Enables GPT text models and `gpt-image-1` for image generation. | [OpenAI API pricing](https://openai.com/api/pricing/) |
| Anthropic | `ANTHROPIC_API_KEY` | Unlocks Claude 3.x text and image generation. | [Anthropic Claude](https://www.anthropic.com/product) |
| Google Gemini | `GEMINI_API_KEY` | Provides Gemini 1.5 models; requires the `google-genai` SDK. | [Gemini API](https://ai.google.dev/) |
| OpenRouter | `OPENROUTER_API_KEY` | Access to community models such as DeepSeek-R1 (free tiers included). | [OpenRouter models](https://openrouter.ai/models) |
| DeepSeek native | `DEEPSEEK_API_KEY` | Direct access to DeepSeek's reasoning models with structured outputs. | [DeepSeek API docs](https://api-docs.deepseek.com/) |
| Groq | `GROQ_API_KEY` | Hosted Llama/Gemma/Qwen models with high-speed inference. | [Groq Cloud](https://console.groq.com/) |
| Together AI | `TOGETHER_API_KEY` | DeepSeek, Llama, Mixtral, and diffusion models with pay-as-you-go pricing. | [Together AI pricing](https://www.together.ai/pricing) |
| Cloudflare Workers AI | `WORKERS_AI_API_TOKEN` | Serverless inference with daily free allowances. | [Workers AI](https://developers.cloudflare.com/workers-ai/) |
| Hugging Face Inference API | `HUGGINGFACE_API_KEY` or `HUGGINGFACEHUB_API_TOKEN` | Text and image endpoints hosted by Hugging Face. | [HF Inference](https://huggingface.co/inference-api) |

## Local and self-hosted runtimes

Serve open-weight reasoning models through an OpenAI-compatible endpoint and
point Coord2Region at it:

- `AI_BASE_URL` (optional): override the default `https://api.openai.com/v1`
  base URL. Use this for vLLM, TGI, Ollama, or other compatible gateways.
- `AI_API_KEY` (optional): pass-through token for bespoke gateways that still
  speak the OpenAI protocol.
- `AI_MODELS` (optional): comma-separated `alias:model_id` pairs to expose
  multiple local models (for example,
  `AI_MODELS=local-r1:deepseek-r1-distill-qwen-14b,local-llama:llama-3.1-70b`).
- `HUGGINGFACE_MODEL_PROVIDERS` (optional): comma-separated mappings assigning
  inference providers to Hugging Face models (for example,
  `HUGGINGFACE_MODEL_PROVIDERS=gpt-oss-120b:sambanova,llama-3.3-70b-instruct:together,deepseek-r1:together,stabilityai/stable-diffusion-xl-base-1.0:replicate`).

Popular self-hosted options include:

- **vLLM** – deploy DeepSeek-R1 distillations or Llama/Gemma models with high
  throughput.
- **Hugging Face TGI** – use the Messages API mode for chat completions.
- **Ollama** – quick local experiments; community images cover many reasoning
  models.

## Configuration workflow

1. Run `python scripts/configure_coord2region.py` and answer the prompts for the
   providers you plan to use.
2. Keep the generated `config/coord2region-config.yaml` private. The file is
   listed in `.gitignore` and loaded automatically by `coord2region.ai_helpers`.
3. (Optional) If you prefer flat files, copy `.env.example` to `.env`, fill in
   the keys you need, and load it with `python-dotenv` or your process manager.

Unset variables – regardless of whether they come from the YAML config, your
shell, or a `.env` file – simply disable their corresponding providers.

## Hugging Face (text + image)

### Create an account and API key

1. Sign up or log in at [huggingface.co](https://huggingface.co).
2. Navigate to **Settings → Access Tokens**, create a *Read* token, and copy it
   to `HUGGINGFACE_API_KEY` (or `HUGGINGFACEHUB_API_TOKEN`).
3. Accept the terms on any gated model card (e.g., Stable Diffusion 3.5) to
   grant the token access.

### Configure providers

Coord2Region’s Hugging Face integration relies on the hub’s Inference API and
router. Router-hosted models must be mapped to a provider you are entitled to
use. Set `HUGGINGFACE_MODEL_PROVIDERS` to a comma-separated list of
`alias:provider` pairs:

```
HUGGINGFACE_MODEL_PROVIDERS=\
  gpt-oss-120b:sambanova,\
  deepseek-r1:together,\
  deepseek-r1-distill-qwen-14b:hf-inference,\
  llama-3.3-70b-instruct:together,\
  stabilityai/stable-diffusion-xl-base-1.0:replicate,\
  stabilityai/stable-diffusion-3.5-large:fal-ai
```

| Alias | Backend ID | Typical provider | Context / latency | Cost & access notes |
| --- | --- | --- | --- | --- |
| `gpt-oss-120b` | `openai/gpt-oss-120b` | SambaNova router | Up to 128k tokens | Free developer tier with daily cap; pro tiers for higher throughput. |
| `deepseek-r1` | `deepseek-ai/DeepSeek-R1` | Together AI router | 128k tokens | Free queue tier; paid for priority and higher rate limits. |
| `deepseek-r1-distill-qwen-14b` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` | HF Inference | 128k tokens | Pay-as-you-go after the free allowance. Ideal for local deployment. |
| `llama-3.3-70b-instruct` | `meta-llama/Llama-3.3-70B-Instruct` | Together AI router | 128k tokens | Limited free credits; requires acceptance of Meta license. |
| `distilgpt2` | `distilgpt2` | HF Inference | 1k tokens | Always free; useful for tests and offline fallback. |
| `stabilityai/stable-diffusion-xl-base-1.0` | SDXL Base | Replicate router | ~10–12 s per image | Low-cost entry point (~$0.01–$0.02 per image). |
| `stabilityai/stable-diffusion-3.5-large` | SD3.5 Large | Fal AI router | ~20 s per image | Paid-only provider; ensure your account has Fal AI billing enabled. |

> **Token limits & pricing.** Router providers can change quotas frequently.
> Always confirm allowances and per-call pricing on the provider’s dashboard
> (SambaNova, Together, Replicate, Fal AI, etc.) before production use.

### Sample usage

```python
from coord2region.llm import generate_summary, generate_region_image

# Text generation
summary = generate_summary(ai, studies, coord, model="gpt-oss-120b")

# Text-to-image (SDXL Base through Replicate)
png_bytes = generate_region_image(
    ai,
    coord,
    region_info,
    model="stabilityai/stable-diffusion-xl-base-1.0",
)
```

The refreshed gallery scripts demonstrate real-world workflows:

- `python examples/ai_text_summary.py` generates atlas labels, fetches nearby
  studies, and produces a concise summary.
- `python examples/ai_reasoned_report.py` builds a narrative plus structured
  JSON using the helpers in `coord2region.ai_reports`.
- `python examples/ai_image_workflow.py` renders an AI image and Nilearn
  fallback, storing artefacts under `ai_examples_outputs/`.
- `python scripts/generate_reasoned_template.py --x 30 --y -22 --z 50`
  runs the full template expected by downstream AI components and saves the
  narrative, JSON payload, structured prompt, and image spec to disk.

If a provider mapping is missing, the call falls back to the generic HF
Inference API, which may reject router-only models.

## OpenAI (text + image)

### Create an account and API key

1. Register at [platform.openai.com](https://platform.openai.com).
2. Add a paid billing method; OpenAI no longer offers unlimited free usage.
3. Generate a secret key under **Settings → API keys** and store it in
   `OPENAI_API_KEY`.

### Supported models in Coord2Region

| Alias | OpenAI catalog ID | Context window | Notes |
| --- | --- | --- | --- |
| `o4` | `o4` | 200k tokens | Latest flagship reasoning model with tool-use aware responses. |
| `o4-mini` | `o4-mini` | 200k tokens | Fastest multi-turn reasoning option; ideal default for summaries. |
| `o3-mini` | `o3-mini` | 200k tokens | Structured reasoning with JSON/thinking traces. |
| `gpt-4.1` | `gpt-4.1` | 128k tokens | Successor to GPT-4; balanced quality and latency. |
| `gpt-4.1-mini` | `gpt-4.1-mini` | 128k tokens | Cost-effective GPT-4.1 tier suitable for production summaries. |
| `gpt-4o` | `gpt-4o` | 128k tokens | Multimodal flagship (text, audio, vision) via Responses API. |
| `gpt-4o-mini` | `gpt-4o-mini` | 128k tokens | Affordable multimodal option and Coord2Region default. |
| `gpt-image-1` | `gpt-image-1` | N/A | DALLE-based image generation; supports 1024×1024 max. |

Consult the [OpenAI pricing page](https://openai.com/api/pricing) for live
token and image costs; tiers evolve frequently.

### Sample usage

```python
from coord2region.llm import generate_summary, generate_region_image

summary = generate_summary(ai, studies, coord, model="gpt-4o-mini")
png_bytes = generate_region_image(ai, coord, region_info, model="gpt-image-1")
```

> **Rate limits.** OpenAI enforces per-minute and per-day budgets based on your
> spend and account age. If you encounter HTTP 429 errors, request higher
> limits in the OpenAI dashboard or upgrade your plan.
