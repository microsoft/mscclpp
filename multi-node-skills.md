# Skill: Running MSCCL++ multi-node tests in containers

How to launch and run any multi-node MSCCL++ test (DSL examples, `mscclpp-test`
performance binaries, `mp_unit_tests`, Python pytest, etc.) across multiple GPU
nodes using bind-mounted dev containers and OpenMPI.

This skill was validated on two H100 nodes using the `ghcr.io/microsoft/mscclpp/mscclpp:base-dev-cuda12.9`
image and a 16-rank (2×8 H100) job. The recipe generalizes to N nodes — just extend
the hostfile and `--np` / `--npernode`.

In what follows, `<NODE_A>`, `<NODE_B>`, … are the hostnames of your nodes, and
`<REPO>` is the host-side path to your `microsoft/mscclpp` checkout (must be the
**same path on every node**, e.g. `/home/azureuser/rjsouza/mscclpp`).

---

## 0. Prerequisites (on the host of *each* node)

- A clone of `microsoft/mscclpp` at the **same absolute path on every node**. The path
  is bind-mounted into the container at `/root/mscclpp`. **The file `setup-ssh.sh`
  must exist at that path on every node** — `docker run -v <file>:<file>:ro` will
  silently create an empty *directory* on any node where the file is missing,
  breaking ssh setup. `scp` it over first if needed.
- An SSH keypair at `~/.ssh/key` (private) usable for password-less access between
  containers (the keypair is reused as the container's own SSH identity by
  `setup-ssh.sh`).
- The repo on each node should ideally be a **full git clone** (with `.git`). If a
  node only has a tarball/snapshot, `pip install -e .` fails because
  `setuptools-scm` can't infer the version — see the workaround in step 3.
- Hostname-based SSH between host accounts already works (`ssh <user>@<peer>`
  succeeds). The MPI launcher reaches the peer container's sshd via this host
  hostname.
- A management interface (we used `eth0`) on every node, used for both MPI OOB
  bootstrap and MSCCL++ `TcpBootstrap`. If your interface differs, substitute it
  everywhere `eth0` appears below.

## 1. Start the dev container on every node

On **each** node:

```bash
cd <REPO>     # directory that contains setup-ssh.sh
docker run -itd \
   --name=rjsouza-mscclpp \
   --privileged \
   --net=host \
   --ipc=host \
   --gpus=all \
   -w /root \
   -v /mnt:/mnt \
   -v $HOME/.ssh:/root/.ssh-host:ro \
   -v $PWD/setup-ssh.sh:/root/setup-ssh.sh:ro \
   -v <REPO>:/root/mscclpp \
   ghcr.io/microsoft/mscclpp/mscclpp:base-dev-cuda12.9 \
   bash
```

Key flags:
- `--net=host` so the container's sshd is reachable on the host's IPs without port mapping.
- `--ipc=host`, `--gpus=all`, `--privileged` for GPU / IB / SHM access.
- `-v $HOME/.ssh:/root/.ssh-host:ro` ships the keypair into the container read-only;
  `setup-ssh.sh` copies it to the right place.
- `-v <REPO>:/root/mscclpp` is the **repo bind mount**, in addition to the
  `setup-ssh.sh` mount, so any source tree (DSL examples, tests, build artifacts)
  is reachable inside the container.

## 2. Bring up sshd inside each container

The provided `setup-ssh.sh` installs `openssh-server`, drops the host key into
`/root/.ssh/id_mscclpp`, configures `sshd` on **port 22000**, and writes a
`~/.ssh/config` that defaults outbound SSH to port 22000.

```bash
# Run inside the container on every node, passing the other nodes as peers:
# On <NODE_A>:
docker exec rjsouza-mscclpp bash -c "PEER_HOSTS='<NODE_B> <NODE_C> ...' bash /root/setup-ssh.sh"
# On <NODE_B>:
docker exec rjsouza-mscclpp bash -c "PEER_HOSTS='<NODE_A> <NODE_C> ...' bash /root/setup-ssh.sh"
# ...
```

The script's last lines should show `-> <peer>: SUCCESS` for every peer. If the
first node prints `Connection refused` for peers that aren't up yet, just re-run
`setup-ssh.sh` once the remaining nodes have finished — the keys are symmetric so
order doesn't matter once every node is up. Verify cross-container SSH in both
directions:

```bash
docker exec rjsouza-mscclpp ssh <NODE_B> hostname    # from A
docker exec rjsouza-mscclpp ssh <NODE_A> hostname    # from B
```

## 3. Build mscclpp inside each container

The `base-dev` image only ships build tooling — no pre-built `libmscclpp.so` and
no Python package. Build on **each** node (parallelizable):

```bash
docker exec rjsouza-mscclpp bash -c '
  cd /root/mscclpp && mkdir -p build && cd build && \
  cmake -DCMAKE_BUILD_TYPE=Release .. && \
  make -j$(nproc) && \
  cd /root/mscclpp && python3 -m pip install -e .
'
```

(Add `-DBYPASS_GPU_CHECK=ON -DUSE_CUDA=ON` etc. as needed for your platform; see
`docs/quickstart.md` for build options. Tests that only need binaries — e.g.
`mscclpp-test`, `mp_unit_tests` — can skip `pip install -e .`.)

**Important gotchas:**

- The image ships a venv at `/root/venv`. `docker exec` inherits the PATH where
  `python3` resolves to `/root/venv/bin/python3`, so `pip install -e .` installs
  `mscclpp` into the venv (not system Python). Remember this in step 6.
- If a node's repo is **not a full git clone** (`.git` missing), `pip install -e .`
  dies in `setuptools-scm` with "unable to detect version". Workaround: get the
  version from a node that *does* have `.git` and pass it via env:
  ```bash
  # on the good node:
  docker exec rjsouza-mscclpp python3 -c "import mscclpp; print(mscclpp.__version__)"
  # -> e.g. 0.9.0.post1.dev24+ga945735b8

  # on the broken node:
  docker exec -e SETUPTOOLS_SCM_PRETEND_VERSION=<that-value> rjsouza-mscclpp \
      bash -c 'cd /root/mscclpp && python3 -m pip install -e .'
  ```
- Smoke check: `docker exec rjsouza-mscclpp python3 -c "import mscclpp; print('OK')"`

## 4. Prepare test artifacts

Whatever your test is, get the inputs onto every node:

- **DSL tests** (`python/mscclpp/language/tests/...`): the DSL program prints the
  JSON execution plan to stdout. Compile it once on the head node:
  ```bash
  docker exec rjsouza-mscclpp bash -c '
    cd /root/mscclpp && \
    python3 <path/to/your_test.py> [args...] > /root/mscclpp/<plan>.json
  '
  ```
  Because the bind mount is per-node, **the JSON written on one node does NOT
  appear on the others.** Copy it:
  ```bash
  for peer in <NODE_B> <NODE_C> ...; do
    scp -i ~/.ssh/key <REPO>/<plan>.json <user>@${peer}:<REPO>/<plan>.json
  done
  ```
  (Alternative: put `<REPO>` on shared storage and skip the copy.)
- **`mscclpp-test` / `mp_unit_tests` binaries**: already in `build/bin/...` after
  step 3 — no copy needed if all nodes ran the build.
- **Python `pytest`**: source files come via the bind mount; no copy needed
  unless you generate intermediate artifacts on the head node.

## 5. Write the hostfile inside the head container

```bash
docker exec rjsouza-mscclpp bash -c 'cat > /root/mscclpp/hostfile <<EOF
<NODE_A> slots=<gpus_on_A>
<NODE_B> slots=<gpus_on_B>
# ... one line per node
EOF'
```

For an 8-GPU-per-node H100 cluster, `slots=8`.

## 6. Launch the test with mpirun

Run from inside the head container. **The flags below are the minimum that
worked across containers** — do not drop them without understanding why (see the
table further down):

```bash
docker exec rjsouza-mscclpp bash -c '
mpirun --allow-run-as-root --bind-to numa \
  -hostfile /root/mscclpp/hostfile \
  -mca btl_tcp_if_include eth0 \
  -mca oob_tcp_if_include eth0 \
  -mca plm_rsh_args "-p 22000" \
  -np <total_ranks> -npernode <gpus_per_node> \
  -x MSCCLPP_DEBUG=WARN \
  -x MSCCLPP_SOCKET_IFNAME=eth0 \
  -x PATH=/root/venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  -x LD_LIBRARY_PATH=/root/mscclpp/build/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
  <YOUR_COMMAND>
'
```

Choose `<YOUR_COMMAND>` based on what you're running:

- **DSL via executor harness:**
  ```
  /root/venv/bin/python3 /root/mscclpp/python/test/executor_test.py \
      -path /root/mscclpp/<plan>.json --size <S> [--in_place] [--dtype float16]
  ```
  `--size` accepts the `K`/`M`/`G` suffix form (`1M`, `256M`, `4G`); for
  collectives it's the **total output buffer size** across all ranks and must be
  divisible by `num_ranks × dtype_size`. Match `--in_place` to the plan's
  `inplace=` flag.
- **C++ perf binary:**
  ```
  /root/mscclpp/build/bin/mscclpp-test/<collective>_test_perf -b 1K -e 1G -f 2 -k <kernel>
  ```
- **Multi-process unit tests:**
  ```
  /root/mscclpp/build/bin/mp_unit_tests -ip_port <head_ip>:20003
  ```
  (Resolve `<head_ip>` from `eth0` on the head node; the C++ binaries need an
  explicit `ip:port`, unlike the Python `executor_test.py` which bootstraps via
  `mpi4py`.)
- **Python pytest harness:**
  ```
  /root/venv/bin/python3 -m pytest /root/mscclpp/python/test/...
  ```

If the run succeeds you'll get per-rank output (timing lines for DSL/perf tests,
`PASS`/`FAIL` for unit tests).

---

## Required flags — and why each matters

| Flag | Reason |
| --- | --- |
| `-mca plm_rsh_args "-p 22000"` | OpenMPI's `orte` launches `orted` over SSH; `setup-ssh.sh` puts sshd on port 22000, not 22. Without this, mpirun tries port 22 (the host's sshd) and `orted` never starts on the peer. |
| `-mca btl_tcp_if_include eth0` / `-mca oob_tcp_if_include eth0` | A typical GPU host has many IPs (docker bridge `172.17.0.1`, IB IPoIB addresses, …). Pinning OMPI's TCP BTL and OOB to `eth0` prevents bootstrap from trying unreachable addresses and stalling. |
| `-x MSCCLPP_SOCKET_IFNAME=eth0` | MSCCL++'s own `TcpBootstrap` is separate from MPI's bootstrap. Without this it may pick a wrong interface and hang in `tryAccept`/connect. |
| `-x PATH=…:/usr/local/cuda/bin:…` | Any test that JIT-compiles CUDA (e.g. `executor_test.py`'s `KernelBuilder` running `nvcc` for the correctness check) needs `nvcc` on PATH. SSH-launched processes don't inherit the interactive shell's PATH. |
| `-x LD_LIBRARY_PATH=/root/mscclpp/build/lib:/usr/local/cuda/lib64:…` | The Python package `dlopen`s `libmscclpp.so` from the build tree; CUDA runtime libs likewise. |
| Absolute `/root/venv/bin/python3` | The image's default venv is **not** auto-activated under `mpirun`'s SSH sessions; bare `python3` resolves to `/usr/bin/python3`, where `mscclpp` isn't installed. Either invoke the venv python directly (preferred) or install `mscclpp` system-wide. |
| `-npernode <gpus_per_node>` | Pins the rank-per-node distribution explicitly. Without it OMPI's default placement can over- or under-subscribe nodes when slots are misread. |
| `--bind-to numa` | Lines up each rank with the NUMA node containing its GPU; mirrors what the in-repo CI deployment scripts use. |

---

## Triage checklist when something goes wrong

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `setup-ssh.sh: Is a directory` | `setup-ssh.sh` was missing on that host at `docker run` time; `-v` made an empty dir. | `sudo rmdir` the empty dir on the host, `scp` the real file, recreate the container. |
| `setuptools-scm was unable to detect version` during `pip install` | Repo on that node has no `.git`. | Set `SETUPTOOLS_SCM_PRETEND_VERSION=<value>` (see § 3) and retry. |
| `ModuleNotFoundError: No module named 'mscclpp'` only when launched by `mpirun` | `mpirun` is invoking system `/usr/bin/python3`, not the venv where `pip install -e .` put `mscclpp`. | Invoke `/root/venv/bin/python3` explicitly in the mpirun command. |
| Job hangs after starting; no output | MSCCL++ bootstrap interface mismatch, or `orted` couldn't SSH to the peer (wrong port). | Re-run with `-x MSCCLPP_DEBUG=INFO -x MSCCLPP_DEBUG_SUBSYS=BOOTSTRAP,NET,INIT`; verify `-mca plm_rsh_args "-p 22000"` and `MSCCLPP_SOCKET_IFNAME=<iface>` are set. |
| `FileNotFoundError: 'nvcc'` from `KernelBuilder._compile_cuda` | CUDA bin missing from PATH under mpirun. | Add `-x PATH=…:/usr/local/cuda/bin:…`. |
| `ExecutorError: Size per chunks inconsistent` | `--size` not divisible by `num_ranks × dtype_size`, or `--in_place` mismatch with the compiled plan's `inplace`, or `-np` doesn't match the plan's rank count. | Match `-np` to the plan's `gpu_size`; pick `--size` accordingly; align `--in_place` with the DSL's `inplace=`. |
| `tryAccept ... Resource temporarily unavailable, retrying` from `TcpBootstrap` (a few times) | Normal — the listening socket is non-blocking; ignore unless it never stops. Persistent retries mean the peer can't reach the head's bootstrap IP — check `MSCCLPP_SOCKET_IFNAME` and routing. |
| `NpKit::Dump ... MSCCLPP library was not built with NPKit enabled` warning | Benign — NPKit profiling instrumentation isn't compiled in by default. Ignore unless you specifically need NPKit traces (then rebuild with `-DNPKIT_FLAGS=...`). |
| Process count mismatch / one node never joins | Hostfile slot count wrong, or `-npernode` and `-np` inconsistent. | `total_ranks = sum(slots)`; `-npernode` must divide `-np`. |
| Stale state between runs | Crashed processes still holding ports / GPU memory. | Inside the container: `ps -eo pid,comm,args \| grep -E 'python\|orted'` then `kill <pid>`. Do this on every node. |

---

## Cleanup

```bash
# On each host:
docker stop rjsouza-mscclpp && docker rm rjsouza-mscclpp
```

Build artifacts under the bind-mounted repo (`build/`, `.eggs/`, etc.) persist on
the host. Either rebuild incrementally on the next run or `rm -rf build` to start
clean.

---

## Validation reference

The procedure above was validated end-to-end with the DSL multi-node AllGather
example (`python/mscclpp/language/tests/multi_node/allgather.py --gpus_per_node 8`)
on 2 × 8 H100 (16 ranks), `--size 1M --in_place`, fp16, yielding ~48.7 µs per
rank with correctness OK.
