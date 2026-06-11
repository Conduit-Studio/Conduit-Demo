# Fleet Predictive Maintenance

This demo shows a Conduit workflow for nightly fleet predictive maintenance. It lists CSV telematics files, maps over each vehicle file, scores maintenance risk, stores vehicle health rows, aggregates critical vehicles, and opens work-order payloads before notifying a depot supervisor.

The example is intentionally structured like a small real Python project. The Conduit entry files under `run_code/` import helper modules from `run_code/fleet/`, so Verify Code and deploy image bundling exercise package import tracing instead of only copying one flat Python file.

## Layout

```text
examples/transportation/fleet-maintenance/
  data/conduit/score-vehicle/input.json
  data/conduit/collect-critical/input.json
  data/conduit/open-workorders/input.json
  data/telematics/2026-06-05/veh-001.csv
  data/telematics/2026-06-05/veh-002.csv
  ...
  run_code/score_vehicle.py
  run_code/collect_critical.py
  run_code/open_workorders.py
  run_code/fleet/
    __init__.py
    io.py
    parsing.py
    risk.py
    summary.py
    workorders.py
  run_code/requirements.txt
  tests/test_run_code.py
```

## Verify Code

After binding this repository and branch to a Conduit workflow, verify each Run Code entry before deploying:

1. Verify `score_vehicle`
   - Entry: `examples/transportation/fleet-maintenance/run_code/score_vehicle.py`
   - Handler: `main`
   - Fixture JSON: `examples/transportation/fleet-maintenance/data/conduit/score-vehicle/input.json`

2. Verify `collect_critical`
   - Entry: `examples/transportation/fleet-maintenance/run_code/collect_critical.py`
   - Handler: `main`
   - Fixture JSON: `examples/transportation/fleet-maintenance/data/conduit/collect-critical/input.json`

3. Verify `open_workorders`
   - Entry: `examples/transportation/fleet-maintenance/run_code/open_workorders.py`
   - Handler: `main`
   - Fixture JSON: `examples/transportation/fleet-maintenance/data/conduit/open-workorders/input.json`

The fixture JSON is the payload passed to `main(inputs)` during the check. The checked-in fixtures point at the hosted demo bucket `try-conduit-app`. If you copy this workflow to your own bucket, update the bucket values in the fixture JSON files and in the S3 List node.

## Conduit Nodes

Use these workflow nodes:

1. Schedule
   - Nightly at 02:00, or `rate(1 day)` while testing schedule deploy behavior.

2. S3 List
   - Bucket: your demo bucket
   - Prefix: `examples/transportation/fleet-maintenance/data/telematics/2026-06-05/`
   - Output: `objects`

3. Map: `score_fleet`
   - Iterate over `objects`.

4. Run Code inside Map: `score_vehicle`
   - Code: `examples/transportation/fleet-maintenance/run_code/score_vehicle.py`
   - Requirements: `examples/transportation/fleet-maintenance/run_code/requirements.txt`
   - Runtime: Container / CPU
   - Input: `object`
   - Output: `vehicle_health`

5. DynamoDB Put JSON Item inside or after Map
   - Table: `fleet-vehicle-health`
   - Item: `vehicle_health`
   - Partition key: `pk`

6. Run Code: `collect_critical`
   - Code: `examples/transportation/fleet-maintenance/run_code/collect_critical.py`
   - Requirements: `examples/transportation/fleet-maintenance/run_code/requirements.txt`
   - Runtime: Lambda or Container / CPU
   - Input: `results`
   - Output: `critical`

7. Choice
   - Branch on `critical.count > 0`.

8. Run Code: `open_workorders`
   - Code: `examples/transportation/fleet-maintenance/run_code/open_workorders.py`
   - Requirements: `examples/transportation/fleet-maintenance/run_code/requirements.txt`
   - Runtime: Lambda or Container / CPU
   - Input: `critical`
   - Output: `workorders`

9. Notify Slack
   - Critical branch: send `critical.summary` and `workorders.message`.
   - Default branch: send an all-clear summary from `critical.summary`.

## Upload Shape

Upload the files under this example to the same S3 key shape:

```text
s3://your-demo-bucket/examples/transportation/fleet-maintenance/data/telematics/2026-06-05/veh-001.csv
s3://your-demo-bucket/examples/transportation/fleet-maintenance/data/telematics/2026-06-05/veh-002.csv
...
```

With AWS CLI:

```bash
aws s3 sync data s3://your-demo-bucket/examples/transportation/fleet-maintenance/data
```

The `data/conduit/.../input.json` files stay in GitHub. They are only used by Verify Code.

## Local Smoke Test

From this directory:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests
```

The smoke test uses local files only. It calls the same `main(inputs)` functions Conduit uses for Run Code.
