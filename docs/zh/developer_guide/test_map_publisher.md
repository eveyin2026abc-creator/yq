# Multi-branch test_map publisher

`publish_test_maps.py` updates one validated map per configured source branch. Start from `scripts/test_map_publisher.example.json`; credentials in URLs are rejected. `--mode generate` needs no token. `--mode publish` uses the server credential. `--mode all` is only for diagnostics.

```bash
python scripts/helpers/ci_gate/publish_test_maps.py --config /etc/msmodeling-test-map/publisher.json --workspace /var/lib/msmodeling-test-map --once
sudo cp contrib/systemd/test-map-generator.service contrib/systemd/test-map-publisher.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now test-map-generator.service test-map-publisher.service
```

Create the users, group, directories, and root-owned configuration named by those units. Store the username and PAT in `/etc/msmodeling-test-map/gitcode-username` and `gitcode-token` with mode `0600`. Each artifact branch contains `test_map.json` at its root. `{branch}` comes from `MSMODELING_TEST_BASE_BRANCH` and is URL-encoded while preserving `/`. Templates must be HTTP(S), contain one `{branch}`, and include no credentials. The default host allowlist is `gitcode.com`; extend it with `MSMODELING_TEST_MAP_ALLOWED_HOSTS`. A missing or stale map still blocks or falls back to the full suite.
