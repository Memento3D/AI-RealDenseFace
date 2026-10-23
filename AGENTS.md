# Development rules

This independent prototype borrows architecture principles from Memento3D-AI-check.
It does not implement or inherit claims about the old Hunyuan pipeline.

- domain uses standard-library typed contracts/ports only.
- application orchestrates through domain ports, not concrete adapters.
- pipelines implement versioned policy and depend on domain only.
- adapters own image I/O, model SDK and local storage. Bind at bootstrap.
- Keep config/provenance and checksums. Never overwrite attempts or change model pins silently.
- Never claim identity, authenticity, production calibration, reconstruction fidelity,
  printability or Windows/GPU performance without corresponding evidence.
- No placeholder success output for unimplemented reconstruction, API or MCP.
- No user images, runs, weights, credentials or environments in code archives/git.
- Run unittest discover -s tests -v and scripts/check_architecture.py after changes.
