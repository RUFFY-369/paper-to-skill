# Paper-to-Skill Meta-Compiler Pipeline

An enterprise-grade orchestration framework that extracts machine learning kernel constraints from papers and compiles them into highly optimized physical kernels.

## Package Layout

```text
paper_to_skill/
├── pyproject.toml              # Centralized package configuration
├── src/                        # Production application code
│   └── paper_to_skill/
│       ├── core/               # Orchestration and supervisor systems
│       └── agents/             # Focused subagents (Parser, Test Engineer, Synthesizer)
├── evals/                      # End-to-end benchmark drivers
├── tests/                      # Evaluation validation test gates
└── sandbox/                    # Containerized execution context
```

## Installation
To install the package locally in editable mode:
```bash
pip install -e .
```
