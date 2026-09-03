# Legacy DroidCam configuration import

This directory contains data-only imports from the previous TechGAR project. No
source code or runtime behavior was copied.

Imported JSON:

- `roi_mask_cam1.json`, `roi_mask_cam2.json`
- `parking_slots_cam1.json`, `parking_slots_cam2.json`
- `two_camera.detector.json`
- `two_camera.shared_m_01.json`
- `two_camera.shared_cm_01.json`
- `two_camera.shared_cm_02.json`

The imported files are JSON-equivalent to their sources. Line endings were
normalized while adding them to this workspace.

## Compatibility decisions

- Both camera images, ROI masks and pixel slot layouts declare `1280 x 720`.
- Each pixel layout contains 24 slots: `E01..F08` for cam1 and `A01..C08`
  for cam2.
- `droidcam_shared_vd_17` and `droidcam_shared_vd_18` explicitly name
  `two_camera.shared_m_01.json` in their historical session metadata.
- `two_camera.shared_cm_01.json` is the later 28/08 configuration mentioned in
  the supplied notes, but no local recording currently proves that it belongs
  to that recording. It is retained as an unassigned bootstrap profile.
- The `droidcam_shared_m_04` recording uses `two_camera.shared_cm_02.json`, as
  recorded in its `session_info.schema2.json` and confirmed by the supplied
  project audit.

## Safety boundary

These files are **not production commissioning evidence**. Both imported shared
calibrations use exactly four correspondences per camera. Four points solve the
eight homography degrees of freedom exactly, so their near-zero residual is an
overfit result rather than an independent accuracy measurement.

The new implementation does not adopt the old `bbox_center` map anchor, old
tracking/identity thresholds, or absolute source paths embedded in the legacy
JSON. The imported detector settings are reference-only. The normalized and
portable entry point is `config/site_manifest.json`.

## Local replay

The new `techgar.demo` adapter decodes both MP4 streams against the recorded
per-camera monotonic timestamps. It converts the declared centimetre world map
to the runtime metre convention and runs the current Stage 1-10 pipeline. The
default 0.5 processing scale is projected back onto the original 1280 x 720
frames for inspection.

```powershell
.venv\Scripts\python.exe -m techgar.demo --dataset droidcam_shared_vd_18
```

Open <http://127.0.0.1:8010>. The orange warning is intentional: the visual
replay can be used to review alignment and behaviour, but the four-point
calibration is still not commissioning evidence.
