# Implemented methods and evidence scope

This is the project's own bounded template-fitting integration, not an
implementation of a published end-to-end hair reconstruction method.

## Source and face

- Existing FLAME-compatible workspace topology is checked: source head and
  two eyeballs. Source geometry scale is set to the declared head+neck size.
- Semantic labels use the existing pinned BiSeNet ONNX parser. Three views
  are parsed for palette/feature paint and silhouette consistency.
- The source and soft_smile options are distinct: soft_smile is a bounded
  smooth corner/cheek displacement, not a muscle model or recovered smile.
- A single shared forward gaze with bounded offsets replaces independent
  pupil guesses. Eyelids remain from the source mesh.
- Skin comes from eroded front semantic skin pixels and the middle luminance
  range; median RGB is compared to eight project swatches with CIEDE2000 in
  CIELAB. Lighting is uncalibrated; this is not recovered albedo or measured
  skin tone. It is not the Monk Skin Tone Scale.
- Lips/brows use softly sampled semantic paint. Optional teeth are a continuous
  painted oral-cap region, with no free-standing tooth geometry.

## Reusable hair

Metadata normalizes orientation and source cavity anchors. Initial anisotropic
scale uses robust upper-head bounds; source cavity roof is anchored below the
head top by a small declared overlap. Lower placement relates to source eye
height and head height; it is an interface prior, not true hair-root detection.

For cage fitting, source-cavity radial rays at 4 height levels × 36 angles
provide candidate controls. Corresponding head radial intersections give
bounded offsets. Low levels sample above the eye region to avoid fitting to
ear/nose protrusions. Periodic angular Gaussian weights (sigma 0.40 radians)
and vertical Gaussian weights (sigma 1.3 mm by default) interpolate radial
offsets onto the whole template. Centerline fade avoids a singular axis.

No hairstyle-specific network, ARAP or full reconstruction benchmark runs in
this branch. Affine is the single optional control. Old research code is kept
for compatibility and is not called by V2's default fitting path.

Runtime custom controls change volume, fringe/nape lift and yaw within explicit
bounds. All configurations still pass geometry guards; bounds do not certify
manufacturability. Score = 1-front_IoU + 0.35*(1-mean(side_IoU)) + 10*front_collision.
This is a proposal score, not identity/beauty recognition. The preferred
expression is an explicit product preset, not chosen by a validated emotion metric.

## Boolean and manufacturing repair

- Float64 Manifold surface union retains the source face surface. Enclosed
  negative cavity shells are intentionally filled for this solid head prototype.
- Tolerances .0001/.001/.003/.006/.01 mm are attempted with float32 STL readback
  checks and a bounded volume-change guard. Tolerance is not printer accuracy.
- Vertex-link fan checks detect pinched vertices that edge-watertightness can
  miss. One hair-only voxel repair at .10 mm may be used for pinches/roundtrip
  trouble: material fill, one 6-neighbor closing and two Taubin iterations.
- Reject repair beyond 8% hair volume change or .30 mm sampled surface error.
  These are experimental budgets, not validated similarity thresholds. No skin,
  face or eye voxel remeshing is performed.
- Core facial surface samples exclude eye sockets/lateral temples, protect
  against boolean drift greater than .02 mm relative to the posed source.
  This is not a full Hausdorff or identity-preservation certificate.
- Final STL is read back, checked for edge/vertex manifoldness, consistent
  winding, positive volume and one material component. Euler may be non-2 for
  legitimate handles; no genus-0 rule is imposed.

The reported hair/whole-head normal-ray first-exit chords are sampled proxies,
not exact minimum wall thickness. Self-intersection, mechanical strength and
physical printing are not independently verified. Experimental supports use
coarse voxel layer connectivity and collision tests; no real machine slicer runs.

## Color encoding

Palette/paint inputs are sRGB design values. GLB vertex colors are converted
to linear RGB; eye PNGs remain sRGB. The CPU preview decodes textures, shades
in linear light and encodes the output to sRGB. It is an approximate renderer,
not a calibrated material/illumination simulation or physical color guarantee.

## Primary references and distinctions

- FLAME: https://flame.is.tue.mpg.de/ — existing head representation; V2 does
  not retrain or independently validate the upstream reconstruction.
- BiSeNet parser source: https://github.com/yakhyo/face-parsing — pinned ONNX
  SHA256 remains 0d9bd318e46987c3bdbfacae9e2c0f461cae1c6ac6ea6d43bbe541a91727e33f.
- CIEDE2000: https://www.cie.co.at/publications/colorimetry-part-6-ciede2000-colour-difference-formula
  — palette color-difference calculation, not skin albedo estimation.
- glTF: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html — linear vertex
  color and sRGB base-color texture conventions.
- Manifold: https://github.com/elalish/manifold — surface Boolean backend;
  the tolerance, repair, field guards and evidence reporting are our integration.
- Disney 2014: https://studios.disneyresearch.com/2014/07/27/capturing-and-stylizing-hair-for-3d-fabrication/
  — motivation for stylized physical hair, not a port of their reconstruction.
- Chen et al. 2022: https://doi.org/10.1016/j.gmod.2022.101135 — related printable
  hair from strands. V2 does not generate strands or implement their algorithm.

No claim of new scientific novelty follows solely from combining these parts.
