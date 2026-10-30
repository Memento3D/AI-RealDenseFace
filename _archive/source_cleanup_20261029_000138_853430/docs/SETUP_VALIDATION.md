# Setup update validation

- Existing suite: 143 tests passed (Linux Python 3.12, temporary OpenCV/psutil dependencies).
- Architecture boundaries: PASS.
- Root requirements: pip dry-run --no-deps parsed project/editable and nested requirement paths. This is not a dependency resolver/install validation.
- Windows patch: tested on pinned upstream checkout; repeated application is idempotent. Synthetic checks cover CRLF preservation and refusal on wrong commit/unrecognized source.
- New Python scripts: compiled and help checked.
- Fresh Windows Python 3.11 / CUDA 12.6 installation, extension build and GPU inference: NOT tested here. Existing user's successful runs do not prove this new baseline installs from scratch.
- No reconstruction algorithm changes. Requirements are direct baseline pins/ranges, not a complete environment lock.
