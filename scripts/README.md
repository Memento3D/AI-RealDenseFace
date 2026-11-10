# Scripts and entrypoints

Run commands from the repository root unless the individual tool documents otherwise.

| Folder | Responsibility |
|---|---|
| `api/` | Start and test the API; assembly/demo/base-only remain separate policies |
| `setup/` | Environment installation; never run implicitly during a refactor |
| `experiments/` | Launch versioned head/hair/TripoSR experiments |
| `review/` | Local review servers and probe review |
| `maintenance/` | Asset verification and explicit maintenance operations |

Examples:

```powershell
.\START-HEADAPI-ASSEMBLY.ps1 -CheckOnly
.\scripts\api\START-HEADAPI-ASSEMBLY.ps1
.\scripts\api\START-HEADAPI-DEMO.ps1 -CheckOnly
.\scripts\setup\SETUP-HEADAPI.ps1
.\scripts\experiments\RUN-HEAD24.ps1 -CheckOnly
```

The root assembly launcher is a compatibility entrypoint only. Canonical launcher implementations live here and derive the repository location from their own location, not the terminal working directory.

Python files directly under `scripts/` remain at their public import/CLI paths. They contain mutually referenced versioned implementations, cache producer paths and checksum dependencies. They are not disposable output. Moving them requires a separate import/cache migration with compatibility checks; this layout change does not rewrite algorithm packages.

Business/domain code remains in `src/`; model SDK and image/mesh I/O belong to adapters. `tools/headapi/`, `tools/customization/`, `tools/headlab26/` and `tools/triposr_probe/` retain their module paths and `python -m` entrypoints.
