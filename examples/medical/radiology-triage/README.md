# Medical Radiology Triage

This demo shows a small Conduit workflow for radiology triage. It reads a study manifest, extracts DICOM image features, loads a RandomForestClassifier model, and emits urgency JSON that can drive a Choice branch, DynamoDB write, and Slack notification.

This is a demo workflow, not a clinical model.

## Layout

```text
examples/medical/radiology-triage/
  data/conduit/read-study/input.json
  data/conduit/infer-urgency/input.json
  data/studies/incoming/*.json
  data/studies/raw/ST-SIIM-*/s1/*.dcm
  model/pneumothorax_rf.joblib
  run_code/read_study.py
  run_code/infer_urgency.py
  run_code/requirements.txt
  tests/test_run_code.py
```

## Verify Code

After binding this repository and branch to a Conduit workflow, verify each Run Code entry before deploying:

1. Verify `read_study`
   - Entry: `examples/medical/radiology-triage/run_code/read_study.py`
   - Handler: `main`
   - Fixture JSON: `examples/medical/radiology-triage/data/conduit/read-study/input.json`

2. Verify `infer_urgency`
   - Entry: `examples/medical/radiology-triage/run_code/infer_urgency.py`
   - Handler: `main`
   - Fixture JSON: `examples/medical/radiology-triage/data/conduit/infer-urgency/input.json`

The fixture JSON is the payload passed to `main(inputs)` during the check. The checked-in fixtures point at the hosted demo bucket `try-conduit-app`. If you copy this workflow to your own bucket, update the bucket values in the fixture JSON files and in the model Config JSON.

## Conduit Nodes

Use these workflow nodes:

1. S3 Event
   - Bucket: your demo bucket
   - Prefix: `examples/medical/radiology-triage/data/studies/incoming/`
   - Suffix: `.json`
   - Output: `manifest`

2. Config JSON
   - Output: `model`
   - Value:

```json
{
  "bucket": "your-demo-bucket",
  "key": "examples/medical/radiology-triage/model/pneumothorax_rf.joblib"
}
```

3. Run Code: `read_study`
   - Code: `examples/medical/radiology-triage/run_code/read_study.py`
   - Requirements: `examples/medical/radiology-triage/run_code/requirements.txt`
   - Input: `manifest`
   - Output: `features`

4. Run Code: `infer_urgency`
   - Code: `examples/medical/radiology-triage/run_code/infer_urgency.py`
   - Requirements: `examples/medical/radiology-triage/run_code/requirements.txt`
   - Inputs: `features`, `model`
   - Output: `urgency`

5. Choice
   - Branch on `urgency.class == "critical"` for critical findings.

6. DynamoDB Put JSON Item and Notify Slack
   - Store or notify from the `urgency` JSON.

## Upload Shape

Upload the files under this example to the same S3 key shape:

```text
s3://your-demo-bucket/examples/medical/radiology-triage/data/studies/incoming/study-0001.json
s3://your-demo-bucket/examples/medical/radiology-triage/data/studies/raw/ST-SIIM-0001/s1/000000.dcm
s3://your-demo-bucket/examples/medical/radiology-triage/model/pneumothorax_rf.joblib
```

The `data/conduit/.../input.json` files stay in GitHub. They are only used by Verify Code.

With AWS CLI:

```bash
aws s3 sync data s3://your-demo-bucket/examples/medical/radiology-triage/data
aws s3 cp model/pneumothorax_rf.joblib s3://your-demo-bucket/examples/medical/radiology-triage/model/pneumothorax_rf.joblib
```

## Local Smoke Test

From this directory:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests
```

The smoke test uses local files only. It calls the same `main(inputs)` functions Conduit uses for Run Code.
