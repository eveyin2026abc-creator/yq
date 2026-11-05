# Multi-branch test_map publisher

`publish_test_maps.py` watches an explicit list of source branches, serially
updates their maps with the existing `run_test_map_sync.sh --once` workflow, and
pushes each validated map to its configured artifact branch. It is intended for
a persistent Linux worker, so dependency environments and the uv cache survive
between cycles.

Copy `scripts/test_map_publisher.example.json` outside the source checkout and
edit the source/publish branch pairs. Branch names must be exact; wildcards are
not expanded. Credentials in repository URLs are rejected.

Run one diagnostic cycle before enabling the service:

```bash
python scripts/helpers/ci_gate/publish_test_maps.py \
  --config /home/yinqian/test-map-publisher.json \
  --workspace /home/yinqian/test-map-publisher \
  --once
```

The workspace owns isolated source, virtual environment, artifact checkout and
durable local map state per source branch. Generation state is under
`generation/`; Git publication checkouts are under `publication/`. A source
HEAD is pinned before generation. A generated map is published only after its
schema, regular-file type, non-empty map and `built_from_commit` are validated.
Writes use an atomic regular-file replacement and reject symbolic links.

`--mode generate` fetches and executes monitored source branches but never
opens or pushes the artifact repository. `--mode publish` validates maps and
pushes them without checking out or executing monitored source code. `--mode
all` remains available for a one-off diagnostic run without credential
isolation. Discovered and configured source/publication branch names are
checked together for collisions before any branch is processed.

When `dependency_index` is configured, the worker temporarily resolves the
lock against that registry. It accepts the rewritten lock only when every
package version, artifact filename, SHA-256 digest and size remains identical;
the original lock is restored before source code runs.

For a long-running worker, use separate unprivileged identities. Only the
publisher receives the GitCode credentials; the generator has write access
only to generation state and the uv cache. The following illustrates the host
setup (adapt installation paths as needed):

```bash
sudo groupadd --system msmodeling-test-map
sudo useradd --system --no-create-home --gid msmodeling-test-map --shell /usr/sbin/nologin msmodeling-map-generator
sudo useradd --system --no-create-home --gid msmodeling-test-map --shell /usr/sbin/nologin msmodeling-map-publisher
sudo install -d -o msmodeling-map-generator -g msmodeling-test-map -m 0750 \
  /var/lib/msmodeling-test-map/generation /var/lib/msmodeling-test-map/uv-cache
sudo install -d -o msmodeling-map-publisher -g msmodeling-test-map -m 0750 \
  /var/lib/msmodeling-test-map/publication
sudo install -d -o root -g root -m 0700 /etc/msmodeling-test-map
sudo install -o root -g root -m 0644 scripts/test_map_publisher.example.json \
  /etc/msmodeling-test-map/publisher.json
sudo install -o root -g root -m 0644 contrib/systemd/test-map-publisher.gitconfig.example \
  /etc/msmodeling-test-map/gitconfig
# Edit /etc/msmodeling-test-map/gitconfig to use the publisher's real commit identity.
printf '%s\n' '<GitCode username>' | sudo tee /etc/msmodeling-test-map/gitcode-username >/dev/null
sudo sh -c 'umask 077; systemd-ask-password "GitCode PAT:" > /etc/msmodeling-test-map/gitcode-token'
sudo chmod 0600 /etc/msmodeling-test-map/gitcode-username /etc/msmodeling-test-map/gitcode-token
sudo cp contrib/systemd/test-map-generator.service /etc/systemd/system/
sudo cp contrib/systemd/test-map-publisher.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now test-map-generator.service test-map-publisher.service
journalctl -u test-map-generator.service -u test-map-publisher.service -f
```

Enter the PAT only in the server terminal; never commit it or paste it into a
chat. The example units expect a root-owned runner checkout at
`/opt/msmodeling-test-map-publisher`. The generator and publisher share only
validated map files through the `msmodeling-test-map` group. systemd protects
the host filesystem, home directories and credential mount from the generator.

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
