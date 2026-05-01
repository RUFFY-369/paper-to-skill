# Paper-to-Skill Meta-Compiler Pipeline

An enterprise-grade meta-compiler designed to autonomously extract math formulations and tensor requirements from technical AI papers and output fully functional, high-performance Triton kernels crystallized as **agentskills.io** compliant skill packages.

---

## 🚀 Architectural Design & Features

1. **Stage A.5: Structural Chunking and Routing**
   - Automatically breaks long PDF and markdown technical manuals by headers.
   - Aggressively strips out irrelevant text (e.g., References, Related Work, Acknowledgements) and passes only the math formulations, architecture definitions, and constraints directly to the reasoning parser to prevent the "Lost in the Middle" context degradation phenomenon.

2. **Zero-Trust RESTRICTED-AST Execution Sandbox**
   - Implements a strict `RestrictedPython` and exact-match AST inspection pass to check generated code before execution.
   - Blocks unauthorized system modules and calls (`os.`, `subprocess.`, `eval`, `exec`, etc.) to prevent host escape vulnerabilities.

3. **The Filesystem Handshake & Progressive Disclosure**
   - Directly encapsulates validated kernel operators into directory-based standalone modules ready for direct filesystem ingestion.
   - Employs strict **YAML frontmatter** definitions at the top of the skill markdown to allow for automated Hermes SkillRegistry indexing and dynamic progressive context loading.

---

## 🔒 Security Configuration: Unsafe Bare-Metal Guardrail

Executing untrusted, AI-generated code directly via a raw subprocess on the host machine is a fatal security flaw. To opt-in to bare-metal direct execution acceleration on your trusted cloud GPU clusters (e.g., Vast.ai instances), you must authorize it explicitly:

```bash
export ALLOW_DANGER_RUN_BARE_METAL=true
```

> [!WARNING]
> Activating the `ALLOW_DANGER_RUN_BARE_METAL` flag bypasses default host isolation. Exercise caution and verify generated scientific code before execution.

---

## 🛠️ Installation & Setup

Ensure the correct system dependencies (`torch`, `triton`, `pynvml`) are installed. To set up the meta-compiler in editable mode:

```bash
pip install -e . --break-system-packages
```

---

## 💻 Standardized CLI Compilation & Deployment

The meta-compiler operates as a clean system tool directly via Python:

### 1. Compile a Technical Paper into a Skill Directory
```bash
python -m paper_to_skill compile --pdf sageattention2_raw.txt --target sm_86 --out ./skills/
```

### 2. Install a Generated Skill to the Hermes Skill Registry
This command carries out the filesystem handshake, moving the output folder to the Hermes storage path:
```bash
python -m paper_to_skill install --dir ./skills/sageattention-2/
```
*Alternatively, you can decouple the registries entirely by adding the compiled skill folder to the `external_dirs` array within your `~/.hermes/config.yaml`.*
