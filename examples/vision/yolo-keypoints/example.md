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
  data/conduit/preprocess-image/input.json
  data/conduit/run-keypoints/input.json
  data/conduit/render-results/input.json
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
- `data/processed/images/person-walk-001-processed.png`: one seeded preprocessed image so `run_keypoints.py` can be verified independently.
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


## Real YOLO Weights

The current smoke-test entry keeps inference deterministic so the repo can run without downloading a model during unit tests. For the real workflow, store YOLO pose weights in the user's S3 bucket and pass them to `run_keypoints` as a normal model ref:

```json
{
  "bucket": "your-demo-bucket",
  "key": "examples/vision/yolo-keypoints/model/yolo11n-pose.pt"
}
```

The real `Map[Run Code]: detect_keypoints` node should receive two things:

- the per-item image ref from the Map item; and
- the shared `model`/`weights` S3 ref from Config JSON or a static config input.

The COCO JSON and YOLO labels are ground truth for validation/training. They should not be used as fake prediction output once the real weights path is enabled.

## Verify Code

Bind the workflow to `Conduit-Studio/Conduit-Demo`, then verify each entry with the matching fixture:

1. `preprocess_image`
   - Entry: `examples/vision/yolo-keypoints/run_code/preprocess_image.py`
   - Handler: `main`
   - Fixture JSON: `examples/vision/yolo-keypoints/data/conduit/preprocess-image/input.json`
   - Output port: `preprocessed`

2. `run_keypoints`
   - Entry: `examples/vision/yolo-keypoints/run_code/run_keypoints.py`
   - Handler: `main`
   - Fixture JSON: `examples/vision/yolo-keypoints/data/conduit/run-keypoints/input.json`
   - Output port: `predictions`

3. `render_results`
   - Entry: `examples/vision/yolo-keypoints/run_code/render_results.py`
   - Handler: `main`
   - Fixture JSON: `examples/vision/yolo-keypoints/data/conduit/render-results/input.json`
   - Output ports: `overlay`, `labels`, `preview`

The fixture JSON files point at `try-conduit-app` for the hosted demo. If you use another bucket, update the bucket values before verifying.

## Build in Conduit

Use Run, not Deploy, for this example.

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
   - Input: `object` or `image` from the Map item.
   - Output: `preprocessed`.
   - Static params through Config JSON or node inputs:
     - `output_bucket`: your demo bucket.
     - `output_prefix`: `examples/vision/yolo-keypoints/data/processed/images/`.

4. `Map: detect_keypoints`
   - Iterate over `preprocess_images.results`.
   - Body node: `Run Code` using `run_keypoints.py`.
   - Runtime: Container / CPU.
   - Input: `image` or `preprocessed` from the Map item.
   - For the deterministic smoke-test entry, provide `annotations` as an S3 ref. For the real YOLO entry, provide `model` or `weights` as an S3 ref instead:

```json
{
  "bucket": "your-demo-bucket",
  "key": "examples/vision/yolo-keypoints/model/yolo11n-pose.pt"
}
```

   - Output: `predictions`.
   - Static params:
     - `output_bucket`: your demo bucket.
     - `output_prefix`: `examples/vision/yolo-keypoints/data/predictions/`.

5. `Map: render_overlays`
   - Iterate over the detection results.
   - Body node: `Run Code` using `render_results.py`.
   - Runtime: Container / CPU.
   - Inputs: the processed image ref and prediction ref from upstream outputs.
   - Outputs: `overlay`, `labels`, `preview`.
   - Static params:
     - `output_bucket`: your demo bucket.
     - `output_prefix`: `examples/vision/yolo-keypoints/data/outputs/overlays/`.

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

The `data/conduit/.../input.json` files stay in GitHub. They are only Verify Code fixtures.

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
