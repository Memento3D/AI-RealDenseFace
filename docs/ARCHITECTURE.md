# Architecture and current scope

Inspired by the uploaded Memento3D-AI-check AGENTS.md, domain ports, application
attempt lifecycle and bootstrap bindings. No old implementation or GPU environment copied.

- domain: standard-library dataclasses, Protocols, errors; no SDK/I/O.
- application: attempt/report lifecycle through ports, imports domain only.
- pipelines/photo_set_validation_v1: policy using typed inspector evidence; imports domain only.
- adapters: Pillow/NumPy/MediaPipe CPU inspection and atomic local report storage.
- bootstrap/settings: CLI composition, input/output admission, strict config and resource cleanup.

No runtime package or queue/API placeholders are created. Version 0.2 adds an
external MICA worker with a separate interpreter; it is not a production service. CLI/API should call the same application use case.

## Meaning of results

NEEDS_RETAKE: one or more automatic checks rejected the set.
NEEDS_HUMAN_REVIEW: basic automatic checks passed; NOT identity/authenticity approval.
FAILED: execution/config/model/decode error, not a quality verdict.
CANCELLED: interrupted attempt.

No ACCEPTED/RECONSTRUCTION_READY claim is currently implemented. identity_verified,
real_capture_verified and reconstruction_ready are false. No automatic anti-spoof,
identity matching, reliable occlusion/expression verification, calibrated quality
probability, or proof of photographic origin is implemented. Eye gaze is not head pose.

Slot left/right follows capture instructions. v1 checks front yaw, side absolute yaw,
and opposite signed side directions, not semantic left/right assignment. Mirrored
selfies and Euler sign conventions require local calibration. Backend yaw is an
estimate, not calibrated camera metadata. No surface coverage claim.

JPEG/PNG input is read once after a byte-size check and hashed. All measurements
refer to EXIF-oriented coordinates. Normalized images strip original metadata.
Original files stay unchanged. Reports retain source SHA256 plus model SHA256,
package version, config and attempt identity. Source kind is a user declaration,
not a detector. Generated side views are not valid evidence of reconstruction accuracy.

## Future research gate

Single-photo MICA initialization is not a three-view reconstruction pipeline.
Multi-view fitting needs shared identity/shape, per-view camera/pose, landmark
correspondence, robust occlusion handling and a measured objective. MediaPipe
landmark indices cannot be used as FLAME vertex indices. Texture fusion, hair,
rig/morph targets, asset attachment, API, MCP and printable export are not implemented.
Single-front initialization and OBJ/PLY export integration are implemented in 0.2;
GPU end-to-end execution is pending the licensed assets and model environment.

DECA/EMOCA and MICA dependencies require separate licensing review before product
use. FLAME 2023 Open does not automatically make old checkpoints compatible.

## 0.3 research and inspection paths

- SinglePhotoValidationV1 uses the existing Inspector port and frontal thresholds.
- application/experiment_head orchestrates through FrontGate, HeadBackend and
  ExperimentStore ports; it cannot access a concrete model SDK.
- OneShotMediaPipeInspector closes the CPU task before model inference starts.
- The experiment is registered independently of the three-photo workflow.
- MicaBackend keeps legacy preview behavior by default. Only the new experiment
  opts into an additional smooth preview. Existing head_initialization v1 is unchanged.
- render-head is CPU-only; it reads an existing OBJ and creates a separate run,
  records before/after source hash, and never modifies original geometry/report.
- Smooth rendering adds a versioned diagnostic representation, not geometry.
