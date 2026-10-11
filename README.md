# TeachGrade

**TeachGrade: Training-Free Short Answer Scoring with Teacher Evidence**

Jinhao Yan, Zhihong Shen, Shengjie Wang, Yiting Ren, Wentao Zhang, Yilei Wang,
and Peichao Lai

TeachGrade adapts short-answer scoring using teacher-graded examples, without
fine-tuning model weights. This repository contains the TeachGrade implementation,
configuration templates and offline tests. It does not distribute datasets,
student records, competing graders, comparison experiments or the paper PDF.

## Method

1. **Select teacher examples.** Uniformly sample k distinct examples without
   replacement using a fixed seed. Skill and CaseBank use the same teacher set,
   which is excluded from evaluation.
2. **Adapt the task.** Infer a task specification from teacher evidence and
   compile frozen Skill instructions and CaseBank entries.
3. **Form retrieval cues.** A global PreJudge reads the complete answer and
   produces per-step completion estimates.
4. **Retrieve evidence.** Retrieve similar teacher cases around the estimated
   completion level. Historical absolute teacher scores are not exposed to
   FinalJudge as transferable scores.
5. **Score the answer.** FinalJudge uses the current question, reference,
   analysis, task rubric, Skill and retrieved evidence. Preliminary scores
   are not supplied as final-score anchors.

Task adaptation precedes evaluation; frozen memories are reused during inference.

## Installation

Use Python 3.10 or newer in a dedicated environment:

```sh
python -m pip install -r requirements.txt
cp .env.example .env
```

Set your model endpoint in `.env`:

```dotenv
OPENAI_API_KEY=YOUR_API_KEY
OPENAI_BASE_URL=https://your-openai-compatible-endpoint.example/v1
OPENAI_MODEL=YOUR_MODEL_NAME
LLM_TIMEOUT_SECONDS=120
LLM_MAX_RETRIES=2
```

An OpenAI-compatible endpoint can be hosted remotely or served locally, for
example with Qwen. Remote API inference does not require a local GPU. Local
serving requires the hardware needed by the selected model. Shell variables
take precedence over `.env`.

For the client's DeepSeek non-thinking mode, export
`DEEPSEEK_THINKING_MODE=disabled` in the shell; it is not loaded from `.env`.
Use the backend configuration appropriate for your model.

## Your Dataset

Pass your own dataset path with `--dataset`; the file may be anywhere on your
machine. Nothing downloads or bundles a benchmark dataset. Provide one task
per UTF-8 JSONL file, with one annotated answer per line:

| Field | Content |
| --- | --- |
| `id` | Nonempty, unique example ID |
| `question` | Question text |
| `total` | Maximum question score |
| `reference` | Reference answer |
| `analysis` | Question analysis |
| `manual_label` | Teacher's whole-answer score |
| `steps` | Answer steps with `response`, teacher `label` and `errors` |

Teacher scores and error labels provide adaptation evidence. Evaluation labels
are used for metrics, not as model-visible grading evidence.

Supply your task's rubric and error categories in a separate JSON configuration:

```json
{
  "q_id": "my-task",
  "course": "MySubject",
  "question_type": "ShortAns",
  "guideline": "Your task's actual grading rubric.",
  "errors": [
    {"name": "Incorrect concept", "description": "A conceptual error."},
    {"name": "Incomplete answer", "description": "A required point is missing."}
  ]
}
```

[examples/task_config.json](examples/task_config.json) is a fictional template,
not a dataset. Replace its rubric and labels with those for your task. Error
labels should match your annotations. No benchmark-specific rubrics are built in.

## Quick Start

Use `run_teachgrade.py`. Defaults are `k=20` and `seed=42`; selected rows retain
their original order within each split. Replace the paths below with your own.

### Inspect The Split

```sh
python run_teachgrade.py --dataset /path/to/your_answers.jsonl \
  --k 20 --seed 42 --output outputs/my_task --phase split
```

This phase uses only the standard library and makes no model calls. It saves
teacher and evaluation IDs. By default, all non-teacher examples are evaluated.

### Prepare Memories

```sh
python run_teachgrade.py --dataset /path/to/your_answers.jsonl \
  --task-config /path/to/your_task_config.json \
  --k 20 --seed 42 --output outputs/my_task --phase prepare --run
```

This phase makes model calls and freezes task-adapted memories. The evidence
validator requires at least four teacher examples; the paper uses 20.

### Score And Evaluate

```sh
python run_teachgrade.py --dataset /path/to/your_answers.jsonl \
  --task-config /path/to/your_task_config.json \
  --k 20 --seed 42 --output outputs/my_task --phase evaluate --run
```

Evaluation reuses frozen memories, preparing them first if needed. Model phases
require `--run` because calls can incur charges. The output contains TeachGrade's
own predictions and metrics; no other grader is run.

| Option | Behavior |
| --- | --- |
| `--dataset PATH` | Your annotated JSONL file; required |
| `--task-config PATH` | Your rubric and labels; required for model phases |
| `--k K` | Teacher example count; default 20 |
| `--seed S` | Sampling seed; default 42 |
| `--eval-size N` | Evaluate the first N non-teacher rows |
| `--limit N` | Limit inference without changing teacher selection |
| `--resume` | Resume from saved predictions |
| `--output PATH` | Output directory |

Keep inputs and sampling options consistent across phases. Use a new output
directory when changing data, rubric, split, model configuration or source.
The runner checks split, task configuration and memory hashes before reuse.

## Outputs

- `split_manifest.json`: input hash, seed and split IDs.
- `build_configuration.json`: memory preparation configuration.
- `memory_manifest.json`: frozen-memory hashes and teacher provenance.
- `ksK/`: Skill, CaseBank, TeachGrade predictions, metrics and retrieval records.
- `specifications/`: task-adaptation specifications.
- `private_raw_calls/`: adaptation request and response audit records.

Outputs can contain student answers and model responses. `.env`, `data/` and
`outputs/` are ignored by Git. Files outside those folders are not automatically
protected: review staged files and never publish credentials or unapproved data.

## Reproducibility And Tests

The release uses seeded random teacher selection, not contiguous last-k
selection. Core adaptation, retrieval and scoring prompts are preserved. A fixed
seed reproduces the split for the same ordered input, not necessarily model
responses or historical paper results. No new benchmark results are claimed.

```sh
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Offline tests use fictional inputs only and make no model API calls. They cover
sampling, split disjointness, task configuration, cache validation, memory
compilation and scoring-context contracts.

## Citation

```bibtex
@misc{yan2026teachgrade,
  title = {TeachGrade: Training-Free Short Answer Scoring with Teacher Evidence},
  author = {Yan, Jinhao and Shen, Zhihong and Wang, Shengjie and Ren, Yiting and Zhang, Wentao and Wang, Yilei and Lai, Peichao},
  year = {2026}
}
```

## License

TeachGrade-authored source code is released under the [MIT License](LICENSE).
Retained metric utilities have their original license and attribution in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [licenses/](licenses/).
Code licensing does not grant rights to external datasets or the paper.
