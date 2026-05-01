# Paper-to-Skill Meta-Compiler Pipeline

An enterprise-grade meta-compiler designed to autonomously extract math formulations and tensor requirements from technical AI papers and output fully functional, high-performance Triton kernels crystallized as **agentskills.io** compliant skill packages.

---

## 🚀 Architectural Design & Features

1. **Stage A.5: Structural Chunking and Routing**
   - Automatically breaks long PDF and markdown technical manuals by headers.
   - Aggressively strips out irrelevant text (e.g., References, Related Work, Acknowledgements) and passes only the math formulations, architecture definitions, and constraints directly to the reasoning parser to prevent context degradation.

2. **Zero-Trust RESTRICTED-AST Execution Sandbox**
   - Implements a strict exact-match AST inspection pass to check generated code before execution.
   - Blocks unauthorized system modules and calls (`os.`, `subprocess.`, `eval`, `exec`, etc.) to prevent host escape vulnerabilities.

3. **V2 Multi-File Synthesizer Architecture**
   - Generates and packages direct directory file trees, separating Triton kernels from high-level PyTorch operator execution modules (`nn_module.py` vs `triton_kernel.py`).

4. **Dynamic Hardware Matrix Routing**
   - Cross-compiles generated target source code by evaluating backward/forward hardware capability matrices (`sm_80`, `sm_86`, `sm_90`).

5. **The Filesystem Handshake & Progressive Disclosure**
   - Directly encapsulates validated kernel operators into directory-based standalone modules ready for direct filesystem ingestion.
   - Employs strict **YAML frontmatter** definitions at the top of the skill markdown to allow for automated Hermes SkillRegistry indexing and dynamic progressive context loading.

---

## 🔒 Security Configuration: Unsafe Bare-Metal Guardrail

Executing untrusted, AI-generated code directly via a raw subprocess on the host machine is a high-risk security action. To opt-in to bare-metal direct execution acceleration on your trusted cloud GPU clusters (e.g., Vast.ai instances), you must authorize it explicitly:

```bash
export ALLOW_DANGER_RUN_BARE_METAL=true
```

> [!WARNING]
> Activating the `ALLOW_DANGER_RUN_BARE_METAL` flag bypasses default host isolation. Exercise extreme caution and manually verify generated code.

---

## 🛠️ Installation & Setup

Ensure the correct system dependencies (`torch`, `triton`, `pynvml`) are installed. To set up the meta-compiler in editable mode:

```bash
pip install -e . --break-system-packages
```

### Configure Endpoint Credentials

Create a `.env` file in the project root directory. This file is parsed by the meta-compiler to access your inference provider or local endpoint:

```text
OPENAI_API_BASE="http://localhost:8000/v1"
OPENAI_API_KEY="your-api-key"
```

*Note: If you're running open-source models locally via **vLLM** (e.g., Llama 3 8B), keep your API Base pointed to your local instance.*

---

## 💻 Standardized CLI Compilation & Deployment

The meta-compiler operates as a clean system tool directly via Python:

### 1. Compile a Technical Paper into a Skill Directory
```bash
python -m paper_to_skill compile --pdf <path_to_paper.txt> --target <sm_86|sm_89|sm_90> --out ./skills/
```

### 2. Install a Generated Skill to the Hermes Skill Registry
This command carries out the filesystem handshake, moving the output folder to the Hermes storage path:
```bash
python -m paper_to_skill install --dir ./skills/<compiled_skill_directory>/
```
*Alternatively, you can decouple the registries entirely by adding the compiled skill folder to the `external_dirs` array within your `~/.hermes/config.yaml`.*
