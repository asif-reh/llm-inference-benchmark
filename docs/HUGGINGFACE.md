# Publishing to Hugging Face

Account: [Ki-Asif](https://huggingface.co/Ki-Asif).
Nothing in this repository uploads for you. `scripts/publish_hf.py` only
prints commands, and `--execute` exits 2.

The static Space is the one to ship. It is HTML, CSS, and one JSON file, so
it runs on free CPU Spaces. The Gradio app under `gradio_space/` is a
fallback if you specifically want `sdk: gradio`.

## What you are publishing

`results/example-synthetic/` is **SYNTHETIC EXAMPLE — NOT A REAL BENCHMARK**.
It is there so the Space renders before you have a real run. If you upload it,
the dataset card and the Space README have to keep that label. Replace the
JSON with a real `summary.json` (where `meta.synthetic` is false) before you
cite a number in a CV or a post.

A real run you might upload later:

```bash
export HF_TOKEN=hf_...
python -m benchlab run --config configs/example-hf.yaml
```

That writes `results/hf-inference-api/`. Those numbers include network time,
cold starts, and shared Inference API capacity. The summary's `limitations`
list says so. Copy `summary.json` into the Space and upload the whole run
directory to the dataset.

## 1. Static Space

Create a Space named `inference-benchmark-lab` under `Ki-Asif`. Choose
**Static** as the SDK. You can do that in the website UI
(`https://huggingface.co/new-space`) with:

- Owner: `Ki-Asif`
- Space name: `inference-benchmark-lab`
- SDK: Static
- License: mit
- Visibility: public

The Space root must contain `index.html`. Upload the **contents** of `space/`,
not the parent folder. From the repository root, with the Hub CLI installed
(`pip install huggingface_hub`, which provides `hf`):

```bash
hf upload Ki-Asif/inference-benchmark-lab space . --repo-type space
```

`space .` means "local directory `space/`, destination `.` in the repo".
The Space root then looks like this:

| File | Role |
| --- | --- |
| `README.md` | YAML frontmatter (`sdk: static`) plus the synthetic warning |
| `index.html` | Page |
| `styles.css` | Styles, no external assets |
| `app.js` | Charts and tables, no build step |
| `data/example-synthetic.json` | Summary the page fetches |

Do not upload `app.py` into this Space. A stray Gradio entrypoint fights a
static SDK.

### Space README frontmatter

This is the block already in `space/README.md`. Copy it if you recreate the
file. `sdk` must be `static`.

```yaml
---
title: LLM Inference Benchmark Lab
emoji: "📈"
colorFrom: yellow
colorTo: gray
sdk: static
pinned: false
license: mit
short_description: Client-observed LLM latency and throughput charts
---
```

Keep a visible line in the README body:

```text
SYNTHETIC EXAMPLE — NOT A REAL BENCHMARK
```

until the JSON you are showing is a real run. Then delete that line and say
which config, model revision, and date the file is from. Link the dataset.

The page also reads `?summary=data/some-other.json` when the value is a
relative path. Absolute URLs are ignored.

## 2. Dataset

Create a dataset `Ki-Asif/inference-benchmark-results`
(`https://huggingface.co/new-dataset`). From the repository root:

```bash
hf upload Ki-Asif/inference-benchmark-results dataset_card/README.md README.md --repo-type dataset
hf upload Ki-Asif/inference-benchmark-results results/example-synthetic example-synthetic --repo-type dataset
```

The first command sets the dataset card. The card states that
`example-synthetic/` is not a measurement. The second command puts
`summary.json`, `summary.csv`, `requests.jsonl`, and the folder README at
`example-synthetic/` in the dataset.

When you have a real run, upload that directory under its own name and do not
leave it looking like the synthetic folder:

```bash
hf upload Ki-Asif/inference-benchmark-results results/hf-inference-api hf-inference-api --repo-type dataset
```

Update the dataset card so a reader can tell which folders are real. Include
the prompt-file SHA-256 from `prompts.sha256`. The prompt text is not inside
the summary. Publish `data/prompts/example.jsonl` only if you are comfortable
with those prompts being public; they are already in this git repo.

`scripts/publish_hf.py` prints the two synthetic commands plus the Space
command. It does not accept a token and it does not call the network.

## 3. Gradio Space, if you need one

Create a **different** Space, or you will overwrite the static one. SDK
Gradio. Upload the contents of `gradio_space/`:

```bash
hf upload Ki-Asif/<gradio-space-name> gradio_space . --repo-type space
```

That directory contains `app.py`, `requirements.txt`, `README.md` with
`sdk: gradio`, and `data/example-synthetic.json`. The app only displays the
JSON. Prefer the static Space for a free CPU host; it has the charts and it
does not boot a Gradio runtime.

## 4. Check before you share the URL

- Space README frontmatter says `sdk: static`.
- Opening the Space shows either the synthetic banner or a real run you can
  explain.
- `meta.synthetic` in the JSON matches what the banner says.
- You can point at N, the percentile method, and the hardware note without
  adding a number that is not in the file.
- The dataset card does not describe the mock folder as an A100, H100, or
  Inference API result.
