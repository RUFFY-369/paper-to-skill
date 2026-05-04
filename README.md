# Paper-to-Skill

Compile ML research papers into optimized Triton kernels for `agentskills.io`.

## Features

*   **Constraint Extraction**: Parses PDFs/markdown to extract math formulations and tensor requirements.
*   **AST Safety Sandbox**: Validates generated code via AST inspection to block unsafe operations (`os`, `subprocess`, etc.).
*   **V2 Multi-File Synthesis**: Separates kernels from PyTorch modules (`triton_kernel.py` vs `nn_module.py`).
*   **Hardware-Aware Routing**: Targets specific GPU architectures (`sm_80`, `sm_86`, `sm_90`).
*   **Hermes Integration**: Generates compliant `SKILL.md` with YAML frontmatter for automatic indexing.

## Security

By default, code execution is restricted. To enable local bare-metal execution (e.g., on cloud GPU instances):

```bash
export ALLOW_DANGER_RUN_BARE_METAL=true
```

> [!WARNING]
> This flag bypasses host isolation. Always verify generated code before execution.

## Setup

```bash
pip install -e .
```

### Environment

Create a `.env` in the project root:

```text
OPENAI_API_BASE="http://localhost:8000/v1"
OPENAI_API_KEY="your-api-key"
```

## Usage

### 1. Compile Paper to Skill
```bash
python -m paper_to_skill compile --pdf paper.txt --target sm_86 --out ./skills/
```

### 2. Install to Hermes Registry
```bash
python -m paper_to_skill install --dir ./skills/sageattention-2/
```

## Verification

Once installed, invoke the skill directly in Hermes:

```bash
hermes chat -q "/sageattention-2 'Run attention forward pass'"
```

### Comparison

| Feature | Baseline LLM | Paper-to-Skill |
| :--- | :--- | :--- |
| **Math Accuracy** | Hallucinated/Placeholders | Extracted from Source |
| **Quantization** | Generic Fallback | INT8/FP8 per paper |
| **Performance** | Suboptimal/Non-runnable | Optimized Triton Kernel |
