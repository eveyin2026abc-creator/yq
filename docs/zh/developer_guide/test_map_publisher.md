# Multi-branch test_map publisher

`publish_test_maps.py` watches an explicit list of source branches, serially
updates their maps with the existing `run_test_map_sync.sh --once` workflow, and
pushes each validated map to its configured artifact branch. It is intended for
a persistent Linux worker, so dependency environments and the uv cache survive
between cycles.

Copy `scripts/test_map_publisher.example.json` outside the source checkout and
edit the source/publish branch pairs. Branch names must be exact; wildcards are
not expanded. Use a Git credential helper or SSH agent for the publication
repository. Credentials in repository URLs are rejected.

Run one diagnostic cycle before enabling the service:

```bash
python scripts/helpers/ci_gate/publish_test_maps.py \
  --config /home/yinqian/test-map-publisher.json \
  --workspace /home/yinqian/test-map-publisher \
  --once
```

The workspace owns isolated source, virtual environment, artifact checkout and
durable local map state per source branch. The worker processes branches
serially and holds an OS lock, so a second instance cannot update the same
workspace. A source HEAD is pinned before generation. A generated map is
published only after its schema, non-empty map and `built_from_commit` are
validated. Failed generation leaves the last good map untouched; failed push is
retried before another generation on the next cycle.

For a long-running worker, adapt `contrib/systemd/test-map-publisher.service`,
then install it as root:

```bash
sudo cp contrib/systemd/test-map-publisher.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now test-map-publisher.service
journalctl -u test-map-publisher.service -f
```

Each artifact branch contains `test_map.json` at its root. CI should choose the
artifact branch from the pull request's target branch, download the file, and
set `MSMODELING_TEST_MAP_PATH`. With the example's `test-map/<source>` naming,
the existing zero-config downloader can instead be redirected before the CI gate:

```bash
export MSMODELING_TEST_MAP_URL_TEMPLATE='https://gitcode.com/eveyin1/msmodeling_pr/-/raw/test-map/{branch}/test_map.json'
python build.py test --suite ci_gate
```

`{branch}` is replaced by `MSMODELING_TEST_BASE_BRANCH`. The template must be an
HTTP(S) URL without embedded credentials. CI must still apply the existing
freshness checks and block or fall back to the full suite when the map is
missing or stale.

Useful options are `--python 3.11`, `--interval 60`, and `--timeout 7200`. The
timeout applies independently to dependency synchronization and map generation.
The service terminates child process groups on timeout or interruption, then
restores the source checkout through the existing sync cleanup handlers.
