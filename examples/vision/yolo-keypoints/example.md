# YOLO Keypoint Detection Run-Only Demo

This demo is a Run-only Conduit workflow for image preprocessing and keypoint detection. It is meant to test the development loop, S3 service nodes, Map, Run Code bundles with helper modules, and visual output preview without requiring a deployed event trigger.

The repo includes a deterministic 100-image COCO 2017 validation sample with matching person-keypoint annotations, plus three tiny synthetic PNGs used by the fast local smoke test. It is not the full COCO dataset. Replace the sample with your own S3 images when you want a larger or domain-specific run.

## Problem

A field operations team receives raw site images from many cameras. Before those images can be inspected or used for analytics, they need to be staged, normalized, scored for human keypoints, and written back as both machine-readable labels and visual overlays.

## Solution

Use Conduit Run to list a batch of raw images, copy them into a prepared bucket, preprocess each copied object, run YOLO-style keypoint detection, and render overlays plus JSON labels. The user can preview images and labels from the canvas while all logs and failures stay in the Output panel.

## Layout

```text
examples/vision/yolo-keypoints/
  example.md
  requirements.txt
  data/conduit/preprocess-image/{input.json, person-walk-001.png}
  data/conduit/run-keypoints/{input.json, person_keypoints_demo.json}
  data/conduit/render-results/{input.json, person-walk-001-processed.png}
  data/raw/images/*.jpg
  data/raw/images/*.png
  data/raw/annotations/person_keypoints_coco_sample.json
  data/raw/annotations/person_keypoints_demo.json
  data/raw/labels/*.txt
  data/yolo-pose.yaml
  data/prepared/images/*.png
  data/processed/images/person-walk-001-processed.png
  data/predictions/person-walk-001-keypoints.json
  run_code/preprocess_image.py
  run_code/run_keypoints.py
  run_code/render_results.py
  run_code/yolo_keypoints/*.py
  run_code/requirements.txt
  tests/test_run_code.py
```

## What the Data Represents

- `data/raw/images/*.jpg`: 100 real COCO 2017 validation images selected for person keypoint annotations.
- `data/raw/images/*.png`: three small synthetic COCO-style person images for fast smoke tests.
- `data/raw/annotations/person_keypoints_coco_sample.json`: filtered COCO keypoint ground-truth file for the 100 downloaded JPEGs. This is not model output.
- `data/raw/labels/*.txt`: YOLO-pose label files derived from the COCO JSON, one file per JPEG.
- `data/yolo-pose.yaml`: dataset YAML for real YOLO pose training/evaluation.
- `data/raw/annotations/person_keypoints_demo.json`: tiny synthetic annotation file for smoke tests and independent Verify fixtures.
- `data/prepared/images/*.png`: source images after the S3 Copy staging step. In a real run these objects are copied by Conduit, not manually created.
- `data/processed/images/person-walk-001-processed.png`: one seeded preprocessed image so `render_results.py` can be verified independently.
- `data/predictions/person-walk-001-keypoints.json`: one seeded prediction so `render_results.py` can be verified independently.


## Refresh the COCO Sample

The committed sample already contains 100 images. To regenerate or replace it from the official COCO host, run:

```bash
python scripts/download_coco_sample.py --limit 100 --force
```

This downloads COCO 2017 validation images with person keypoint annotations and writes:

```text
data/raw/images/*.jpg
data/raw/annotations/person_keypoints_coco_sample.json
```


## What To Use

Use these artifacts for different jobs:

- **Inference model:** store a real YOLO pose weight file in S3, for example `s3://your-demo-bucket/examples/vision/yolo-keypoints/model/yolo11n-pose.pt`.
- **Inference input:** use image S3 refs from `S3 List` / `S3 Copy`.
- **Training or evaluation labels:** use `data/raw/labels/*.txt` and `data/yolo-pose.yaml`.
- **Ground truth record:** use `data/raw/annotations/person_keypoints_coco_sample.json` only as COCO ground truth, not as model output.
- **Smoke-test fixtures:** use `data/conduit/.../input.json` only for the Run Code node's built-in Verify and local tests.

The checked-in `run_keypoints.py` keeps inference deterministic so the repo can run without downloading a model during unit tests. A real YOLO version adds two **file ports** — `image` (the pixels) and `model` (the weights) — and reads them as local paths. There is still no S3 or boto3 in the code; Conduit hands the wrapper the local files:

```python
def main(inputs):
    from ultralytics import YOLO
    model = YOLO(inputs["model"])           # file port -> local path to the .pt weights
    results = model(inputs["image"])        # file port -> local path to the image
    prediction = to_coco_pose(results, image_id=str(inputs["image_id"]))
    out_path = str(Path(tempfile.gettempdir()) / f"{inputs['image_id']}-keypoints.json")
    Path(out_path).write_text(json.dumps(prediction))
    return {"predictions": out_path, "prediction": prediction, "image_id": inputs["image_id"]}
```

Conduit note: `Map[Run Code]` exposes `items` for the per-item value and any additional body inputs as shared inputs — wire the per-item image into the body **file port** `image` and the model weights into the shared body **file port** `model`.

## Run Code uses file ports

The three Run Code modules have **no S3 and no boto3**. They read and write plain **local file paths** via Conduit's file ports:

- A port typed `s3-ref` is a **file port**. For an input, Conduit downloads the object and hands `main` a local path in `inputs[port]`; the code does `open(inputs["image"], "rb")`. For an output, `main` returns a local path under that port name and Conduit uploads it.
- A port typed `json` is an inline value (string/number/dict), as today.

Because the wrapper downloads to a generic temp name, the **original filename is lost**. Any id derived from the filename is therefore passed as the JSON field `image_id`, threaded node-to-node:

- `preprocess_image` derives `image_id` from the `image_name` JSON input and emits it.
- `run_keypoints` and `render_results` take `image_id` as a JSON input (wire it from the upstream node) rather than re-deriving it from a path.

Port shapes:

| Node | File-port inputs | JSON inputs | File-port outputs | JSON outputs |
| --- | --- | --- | --- | --- |
| `preprocess_image` | `image` | `image_name`, `target_size` | `preprocessed` | `image_id`, `metadata` |
| `run_keypoints` | `annotations` | `image_id` | `predictions` | `prediction`, `image_id` |
| `render_results` | `image`, `predictions` (optional) | `image_id`, `prediction` (optional) | `overlay`, `labels` | `summary` |

`render_results` accepts the prediction either inline as the `prediction` JSON dict or as the `predictions` file port (a local path to a prediction JSON); supply one.

## Verify a node

Verify each Run Code node using its built-in **Verify** dropdown (collapsed by default — click to expand). Do not use the standalone **Verify Code** node for this example: it has no ports, so it cannot handle file-port code.

**The #1 gotcha — set the port types.** A Run Code node's input and output ports each have a type. `s3-ref` = a file (Conduit downloads it and hands your code a local path; your code does `open(path)`). `json` = a plain inline value (string, number, dict). **If a file input is mistyped `json`, your code receives the path string instead of the file and fails with `[Errno 2] No such file or directory: '<name>'`.** This is the single most common mistake.

Port types for each node:

| Node | File-port inputs (`s3-ref`) | JSON inputs | File-port outputs (`s3-ref`) | JSON outputs |
| --- | --- | --- | --- | --- |
| `preprocess_image` | `image` | `image_name`, `target_size` | `preprocessed` | `image_id`, `metadata` |
| `run_keypoints` | `annotations` | `image_id` | `predictions` | `prediction`, `image_id` |
| `render_results` | `image`, `predictions` (optional) | `image_id`, `prediction` (optional) | `overlay`, `labels` | `summary` |

**Steps.** Bind the workflow to `Conduit-Studio/Conduit-Demo`. On each Run Code node set the entry `.py` and the port types above. Then open the node's **Verify** dropdown — the fixture path is pre-filled — and click **Verify**. Conduit stages the sample files committed beside each fixture and runs the code in your AWS account. No S3 pre-seeding is needed.

For a **file port**, the fixture value is a **relative path to a file beside the fixture** (same directory; `..` is not allowed) — Conduit stages it and hands the code a local path. JSON ports are inline.

1. `preprocess_image`
   - Entry: `examples/vision/yolo-keypoints/run_code/preprocess_image.py`
   - Handler: `main`
   - Fixture JSON: `examples/vision/yolo-keypoints/data/conduit/preprocess-image/input.json` (file port `image` → `person-walk-001.png` beside it; `image_name` is inline JSON)
   - Output port: `preprocessed`

2. `run_keypoints`
   - Entry: `examples/vision/yolo-keypoints/run_code/run_keypoints.py`
   - Handler: `main`
   - Fixture JSON: `examples/vision/yolo-keypoints/data/conduit/run-keypoints/input.json` (file port `annotations` → `person_keypoints_demo.json` beside it; `image_id` is inline JSON)
   - Output port: `predictions`

3. `render_results`
   - Entry: `examples/vision/yolo-keypoints/run_code/render_results.py`
   - Handler: `main`
   - Fixture JSON: `examples/vision/yolo-keypoints/data/conduit/render-results/input.json` (file port `image` → `person-walk-001-processed.png` beside it; `prediction` is an inline JSON dict)
   - Output ports: `overlay`, `labels`

## Build in Conduit

Use Run, not Deploy, for this example.

### Current runnable smoke-test path

This path uses the checked-in deterministic `run_keypoints.py` and is for verifying Conduit mechanics, not real YOLO inference.

1. `S3 List: list_raw_images`
   - Bucket: your demo bucket.
   - Prefix: `examples/vision/yolo-keypoints/data/raw/images/`
   - Suffix: `.jpg` for the 100-image COCO sample, or `.png` for the tiny smoke-test images.
   - Output: `objects`

2. `Map: copy_to_prepared`
   - Iterate over `objects`.
   - Body node: `S3 Copy`.
   - Source: supplied by the Map item.
   - Destination bucket: choose a bucket from the combo box.
   - Destination key: leave blank to keep each source key.
   - Output: `results`

3. `Map: preprocess_images`
   - Iterate over `copy_to_prepared.results`.
   - Body node: `Run Code` using `preprocess_image.py`.
   - Runtime: Container / CPU.
   - Body input ports and types:
     - `image` — `s3-ref` (file port; wire from the Map item's per-item copied object ref).
     - `image_name` — `json` (wire the object's key/filename from the upstream S3 object so the id is derived correctly; this is the one input the file-port model adds vs. self-S3 code — the wrapper loses the original filename, so the name must travel separately as JSON).
   - Output: `preprocessed` (`s3-ref`) and `image_id` (`json`).

Do not use the COCO annotation JSON as a fake model in the production workflow. It is only ground truth and fixture data.

### Real YOLO path to build next

Use this shape for real YOLO weights:

1. `S3 List: list_raw_images`
   - Output: `objects`

2. `Map[S3 Copy]: copy_to_prepared`
   - Items input: `objects`
   - Output: `results`

3. `Map[Run Code]: preprocess_images`
   - Items input: `copy_to_prepared.results`
   - Body input ports:
     - `image`: `s3-ref` (file port — wire from the per-item copied object ref)
     - `image_name`: `json` — wire the object's key/filename from the upstream S3 object (the wrapper loses the original filename, so the id must travel separately as JSON)
   - Outputs: `preprocessed` (`s3-ref`), `image_id` (`json`)

4. `Config JSON: model_ref`
   - Output: `model`
   - Value:

```json
{
  "bucket": "your-demo-bucket",
  "key": "examples/vision/yolo-keypoints/model/yolo11n-pose.pt"
}
```

5. `Map[Run Code]: run_keypoints`
   - Runtime: Container / CPU, or GPU when the GPU runtime is available for Run.
   - Inputs:
     - `items`: wire from `preprocess_images.results`
     - `model`: wire from `Config JSON.model` (shared input to every iteration)
   - Body input ports:
     - `annotations`: `s3-ref` (file port — the COCO annotations file; Conduit hands the code a local path)
     - `image_id`: `json` — wire from `preprocess_images.image_id` (the node does NOT re-derive the id from a path; it receives it from upstream)
     - `model`: `s3-ref` (file port — Conduit hands the code a local path to the `.pt` weights; do not type this `json`)
   - Output ports: `predictions` (`s3-ref`), `prediction` (`json`), `image_id` (`json`).
   - The code reads the local model and annotations paths, derives the pose from the annotations (deterministic demo) or scores the image (real YOLO), returns the prediction JSON as a local path on its `predictions` file port (Conduit uploads it), and emits a compact `prediction`/`image_id` summary.

6. `Map[Run Code]: render_results`
   - Runtime: Container / CPU.
   - Items input: `run_keypoints.results`
   - Body input ports:
     - `image`: `s3-ref` (file port — wire from `preprocess_images.preprocessed`)
     - `image_id`: `json` — wire from `preprocess_images.image_id` (threaded through `run_keypoints`; do NOT re-derive from a path)
     - `prediction`: `json` — wire from `run_keypoints.prediction`
     - `predictions`: `s3-ref` (optional; supply instead of `prediction` if you prefer the full predictions file)
   - Output ports: `overlay` (`s3-ref`), `labels` (`s3-ref`) — both file ports

## Expected Outputs

For every image, the workflow writes:

- A processed PNG under `data/processed/images/`.
- A keypoint JSON label file under `data/predictions/`.
- A visual overlay PNG under `data/outputs/overlays/`.
- A small rail JSON object containing S3 refs to those artifacts.

Use the node preview eye icon on the canvas to inspect overlay PNGs and JSON labels. Use the Output panel for logs and errors.

## Upload Shape

Upload the demo data to the same S3 key shape:

```text
s3://your-demo-bucket/examples/vision/yolo-keypoints/data/raw/images/000000425226.jpg
s3://your-demo-bucket/examples/vision/yolo-keypoints/data/raw/annotations/person_keypoints_coco_sample.json
s3://your-demo-bucket/examples/vision/yolo-keypoints/data/yolo-pose.yaml
s3://your-demo-bucket/examples/vision/yolo-keypoints/model/yolo11n-pose.pt
s3://your-demo-bucket/examples/vision/yolo-keypoints/data/prepared/images/person-walk-001.png
s3://your-demo-bucket/examples/vision/yolo-keypoints/data/processed/images/person-walk-001-processed.png
s3://your-demo-bucket/examples/vision/yolo-keypoints/data/predictions/person-walk-001-keypoints.json
```

With AWS CLI from this example directory:

```bash
aws s3 sync data s3://your-demo-bucket/examples/vision/yolo-keypoints/data
```

The `data/conduit/.../input.json` files stay in GitHub. They are only Verify fixtures for Run Code nodes and local tests.

## Local Smoke Test

From this directory:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests
```

The local test calls the same `main(inputs)` functions Conduit uses. It reads checked-in files locally and writes outputs to a temporary directory.

## Why This Example Exists

This workflow is intentionally different from the deployed medical and fleet examples:

- It is Run-only.
- It uses S3 List plus Map.
- It uses generic S3 Copy before Run Code.
- It tests a multi-file Python package under `run_code/yolo_keypoints/`.
- It produces visual artifacts that should be inspectable from the preview pane.
