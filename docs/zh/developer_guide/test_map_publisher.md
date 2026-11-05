# Multi-branch test_map publisher

`publish_test_maps.py` serially updates and publishes one validated map per configured or discovered source branch. Start from
`scripts/test_map_publisher.example.json`; credentials in URLs are rejected.

Run one diagnostic cycle before enabling the service:

```bash
python scripts/helpers/ci_gate/publish_test_maps.py --config /etc/msmodeling-test-map/publisher.json --workspace /var/lib/msmodeling-test-map --once
```

Each branch has isolated generation and publication directories. Maps must be regular files with valid schema and pinned commit; writes are atomic.
Alternate dependency indexes may change only registry hosts, never package artifacts. Use `--mode generate` without credentials and `--mode publish`
with the scoped GitCode credential. `--mode all` is intended only for diagnostics.

For continuous operation, create the users, group, directories and root-owned configuration referenced by the two example units. Store the username and
PAT in `/etc/msmodeling-test-map/gitcode-username` and `gitcode-token` with mode `0600`; enter the PAT only on the server.

```bash
sudo cp contrib/systemd/test-map-generator.service /etc/systemd/system/
sudo cp contrib/systemd/test-map-publisher.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now test-map-generator.service test-map-publisher.service
```

Each artifact branch contains `test_map.json` at its root. CI selects the pull request target branch and sets `MSMODELING_TEST_MAP_PATH`.
With the example's `test-map/<source>` naming, redirect the existing downloader before the CI gate:

```bash
export MSMODELING_TEST_MAP_URL_TEMPLATE='https://gitcode.com/eveyin1/msmodeling_pr/-/raw/test-map/{branch}/test_map.json'
python build.py test --suite ci_gate
```

`{branch}` is replaced by `MSMODELING_TEST_BASE_BRANCH`; branch names are URL-encoded while preserving `/`. Templates must be HTTP(S) without credentials.
The default host allowlist is `gitcode.com`; declare custom hosts with `MSMODELING_TEST_MAP_ALLOWED_HOSTS` (comma-separated).
CI still blocks or falls back to the full suite when the map is missing or stale.

Useful options are `--python`, `--interval` and `--timeout`.
