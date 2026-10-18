Recipe: YoungAi DeepSeek-V4.1-Flash-1017 on the DGX Spark

This is the corrected 1017 packaging: DeepSeek's October 17 V4.1 Flash checkpoint
through YoungAi's vq8sh14-q4k-mtpnative recipe, with five domain sidecars solved
against this base, and the two official 1017 Engram tables fetched separately from
DeepSeek. It replaces the older Baekpica packaging on port 8008.

- Contract: recipes/youngai-deepseek-v41-flash-1017.yaml
- Launcher: scripts/ds41flash_1017_install.sh (deploy it as ~/ds41flash_1017_install.sh)
- Artifact: wenzhouwu/YoungAi-DeepSeek-V4.1-Flash-1017 at 993a24e1
- Upstream model: deepseek-ai/DeepSeek-V4.1-Flash-1017 at 1bae0e10
- Engine: Baekpica/ds4-dfm-rs, make cuda-spark

What this recipe buys you

A frontier V4.1-class coding and agentic model resident on one Spark at the native
1,048,576-token context, with the three-tower DSpark drafter inside the base,
vision through a sidecar, a choice of five domain sidecars, and retrieval through
the model's own Engram n-gram tables streaming off NVMe. Nothing else in this
repository reaches that depth with that much model behind it.

What it does not buy you is fidelity. The routed experts are 8-dimensional vector
quantization at about 1.6 bits per weight. On V4.1 Flash this recipe agreed with
the original's next token 78.61 percent of the time on code, 78.52 on English and
69.08 on medicine. That is a deliberate quality drop in exchange for fitting on
one box. No 1017 fidelity figure exists yet, here or upstream.

Why there is a new packaging at all

The older Baekpica packaging shipped the base and the vision sidecar but no domain
sidecars, and its base has a different digest from this one at the same byte size:
9f3afe6a... against 0da4ae2d.... They are different files. This repository ships
the base named with 1017, a vision file named with 1017, and five sidecar
directories whose own manifests name this base. Do not mix the two packagings,
and do not reuse the older base because it is already on disk.

The date also matters where file names do not help. All 48 official 1017 shards
differ from V4.1's, but the two that hold the Engram tables keep their names and
sizes: V4.1's official copies hash to 824db488... and 976330f4..., while the 1017
copies hash to 76a05106... and 9dc51433.... A name-and-size check cannot tell
them apart. The launcher hashes both.

Artifacts

Repository: wenzhouwu/YoungAi-DeepSeek-V4.1-Flash-1017 at
993a24e1ea816c6796c2e6c79bc905383baf1829. Public, not gated.

Base: DeepSeek-V4.1-Flash-1017-vq8sh14-q4k-mtpnative.gguf, 113,556,639,424
bytes, SHA-256 0da4ae2d..., assembled from 40 parts (24 x 2,838,915,986 then
16 x 2,838,915,985). The header, read from the first 16 MB of part01 and parsed
with the engine's own ds41_inspect, reports 73 KV entries, 1000 tensors,
data_pos 4,787,744, alignment 32, deepseek4.context_length 1048576, three MTP
towers with targets 37, 38 and 39, and Engram layers 1 and 14. Its internal
general.name is still DeepSeek V4.1 Flash; the 1017 marker is the repository and
file name.

Vision: DeepSeek-V4.1-Flash-1017.vision.gguf, 971,066,016 bytes, SHA-256
3a8b194a.... The digest is identical to the older packaging's vision file, so
these are the same bytes under a date-bearing name. SHA256SUMS does not list it;
the launcher carries the digest.

Sidecars: five directories of 41.8 MB each, 209,154,581 bytes in total:

- finance: ...-grrb-vqfin41_vqhalf_a_n8192-engine, 73 files, aggregate aacdecd2...
- code: ...-grrb-code_fit_n15360-engine, 70 files, aggregate dba83d09...
- law: ...-grrb-law_fit_n15360-engine, 66 files, aggregate 7f9b451d...
- medicine: ...-grrb-med_fit_n15360-engine, 72 files, aggregate 52e3cdb3...
- science: ...-grrb-sci_fit_n15360-engine, 71 files, aggregate b4fedb1a...

The aggregate is SHA-256 over the sorted digest-and-name lines of every gr_L*.bin
and rb_L*.bin in the directory. The launcher recomputes it from disk; all five
were checked end to end here on 2026-10-18, including a full download of the
finance directory.

Engram: not in the YoungAi repository. The launcher fetches the two official 1017
shards from deepseek-ai/DeepSeek-V4.1-Flash-1017 at 1bae0e10:

- model-00047-of-00048.safetensors, 101,535,150,936 bytes, SHA-256 76a05106...
- model-00048-of-00048.safetensors, 101,537,926,640 bytes, SHA-256 9dc51433...

The default engram directory is new, so the fetch is explicit. If you already have
a verified copy of the official 1017 tables elsewhere, point ENGRAM_DIR at it;
verify still hashes both files. Do not point it at a V4.1 copy: the names and
sizes are identical and only the digest separates them.

Where the pieces go, and why

The base is read exactly once at load and then stays resident in unified memory,
so disk speed stops mattering after the first minutes. It goes on the roomy
volume:

    /mnt/models/ds41flash-1017/DeepSeek-V4.1-Flash-1017-vq8sh14-q4k-mtpnative.gguf

Everything read while the server runs goes on the fast device under
/home/leandro/models on the reference Spark:

    /home/leandro/models/ds41flash-1017/model/              five sidecars + vision
    /home/leandro/models/ds41flash-1017/deepseek-engram/    the two 101.5 GB tables
    /home/leandro/models/ds41flash-1017/logs/               server and watchdog logs

That split is the whole point. The sidecars are applied per layer on every token,
the vision tower is read from disk when an image arrives, and the Engram tables
are read in rows on the tokens that hit an n-gram. The base is not read again
after the load.

Quick start

Deploy the script, then run it on the Spark as the user who owns the directories:

    scp scripts/ds41flash_1017_install.sh leandro@192.168.1.91:~/ds41flash_1017_install.sh
    ssh leandro@192.168.1.91
    bash ~/ds41flash_1017_install.sh preflight
    bash ~/ds41flash_1017_install.sh download     # 317.8 GB, resumable
    bash ~/ds41flash_1017_install.sh assemble     # 40 parts -> the base
    bash ~/ds41flash_1017_install.sh verify
    bash ~/ds41flash_1017_install.sh build
    bash ~/ds41flash_1017_install.sh start

install does all six in order. plan, stop, status, logs and test round it out.
Every path, port, domain and memory knob is an environment override documented in
the script header.

If the older 1017 launcher is still running on 8008, stop it with its own stop
command first. The new start refuses while another ds4-server is running, because
the engine takes a single global lock.

Domains

One sidecar at a time, chosen at start. The launcher takes the domain as a flag
or as an environment variable, exactly like the older installer did:

    bash ~/ds41flash_1017_install.sh --domain finance start     # default
    bash ~/ds41flash_1017_install.sh --domain code start
    bash ~/ds41flash_1017_install.sh --domain law start
    bash ~/ds41flash_1017_install.sh --domain medicine start
    bash ~/ds41flash_1017_install.sh --domain science start
    bash ~/ds41flash_1017_install.sh --domain none start        # bare base
    DOMAIN=code bash ~/ds41flash_1017_install.sh start          # same thing

A sidecar rescales routed-expert rows and nudges the router. It adds no layers and
is not the model. The engine does not fingerprint a zchain against the base, so
the directory aggregate is what catches a wrong or corrupted sidecar.

Command line

The flag surface mirrors the older V4.1 installer and adds the fetch commands:

    --dir DIR            fast-device root (sidecars, vision, logs, pid files)
    --gguf-dir DIR       where the base GGUF alone lives
    --engram-dir DIR     the two 101.5 GB tables
    --domain NAME        finance | code | law | medicine | science | none
    --host ADDR          listen address
    --port N             listen port
    --mtp-mode MODE      off | auto | on
    --mem-budget-mb N    the engine load budget
    --min-p F            exported as DS4_SERVER_DEFAULT_MIN_P (see below)
    --engine-dir DIR     the ds4-dfm-rs tree
    --server-bin PATH    the binary to run
    --hf-bin PATH        the Hugging Face CLI
    --revision SHA       pin another model revision
    --watchdog 0|1       memory watchdog
    --no-vision          text only
    --no-build           never run make
    --no-start           everything except starting
    --force-build        make cuda-spark even when a binary exists
    --allow-shared       overrule the single-owner probe
    --posttrain          recognised and refused; this packaging ships none

Every flag also exists as an environment variable of the same name, so
DOMAIN=code and --domain code are the same thing.

Thinking and sampling

YoungAi reports that this weight recipe under-weights stop decisions, so a long
thinking segment can fail to close. The server defaults to non-thinking mode and
thinking is opt-in per request.

The launcher sets no temperature anywhere. Its smoke test sends temperature 0,
the older V4.1 installer's setting, because the check wants a deterministic reply.
Real requests should carry DeepSeek's recommendation for 1017, temperature 1.0
with top_p 1.0, sent explicitly, because sampling follows the request.

The context is not a flag. The GGUF metadata carries 1,048,576 and the engine
takes it from there. Do not add -c or --ctx.

The launcher accepts --min-p and exports DS4_SERVER_DEFAULT_MIN_P, for parity
with the older V4.1 installer. On the engine build now on the box (ds4-dfm-rs at
416d4f5a) that variable feeds the GLM-5.3 family's defaults (parse.rs:132,
380-384); the V4.1 route sets engine_defaults, so its request-omitted min_p is
0.0 (parse.rs:582-587). The value is recorded and passed but does not change this
model's sampling. An explicit min_p in a request still wins.

Ports

8008 is this artifact. 8000 is the V4.1 port deployment, 8003 Qwen3.8, 8005
Bonsai, 8006 YoungAi V4.1, 8007 GLM-5.3 VQ and 8009 GLM-5.3 Uncensored VQ. Only
one ds4 model can be resident at a time whatever the port says.

What a start looks like

The launcher prints the resolved base, sidecar, Engram and vision paths, starts
the server under a watchdog, waits for /v1/models, checks the post-load memory
floor and runs one chat request. The card reports about two minutes to map the
113.6 GB base; the launcher allows twenty minutes before it gives up. Expect the
boot line to report the resolved 1M context and the drafter state. The watchdog
stops the server if MemAvailable stays below 2,500 MB across two five-second
samples.

What verify checks

- the assembled base against SHA256SUMS and the published 0da4ae2d digest
- the vision file against 3a8b194a, which SHA256SUMS does not list
- all five sidecar directories against their aggregates
- both Engram shards against the official 1017 digests, which takes a few minutes
  because the files are 101.5 GB each

What has not been measured

Nothing in this recipe has been loaded, requested or timed on the Spark. Every
speed figure in the contract is V4.1 Flash's, measured with the same recipe and
runtime. The assembled base digest was not reproduced here because that requires
the full 113.6 GB; the header evidence comes from part01, and the two Engram
digests come from the official repository's LFS metadata. The header check used a
sparse full-size image with only the first 16 MB filled, so it proves the header,
the 1000-tensor inventory and every tensor's offset and size; it does not inspect
the weight bytes.

Limits worth repeating

- Lossy at roughly 1.6 bits per routed-expert weight; 1017 fidelity is unmeasured.
- The Engram tables are not interchangeable between V4.1 and 1017.
- The sidecars are not cryptographically bound to the base; the manifest name is
  the provenance and the aggregate is the integrity check.
- The base's internal name does not say 1017. Never identify it by that string.
- One ds4 model at a time.

Troubleshooting

Port 8008 already answers /v1/models. Stop the older launcher first. If you truly
want two owners, ALLOW_SHARED=1 overrules the probe, but only one model fits.

A sidecar aggregate mismatch. Delete that directory and re-run download. A file
from the V4.1 set would produce exactly this symptom.

An Engram digest mismatch. The two names are shared with V4.1, so a mismatch
almost always means the wrong checkpoint's tables. Delete both files and re-run
download to fetch the official 1017 copies.

The server exits while loading. Read logs. The usual causes are a missing Engram
directory (the GGUF records absolute paths from the converting machine, and
--engram-dir swaps only the directory), a short base assembly, or less than
100,000 MB of available memory before the load.

Preflight refuses a base directory under /mnt when it resolves to the root
filesystem. That means /mnt/models is not mounted, and the 113.6 GB base would
land on the system disk. Mount the volume and re-run preflight.

An image request returns HTTP 400. The vision sidecar was not mounted. Start
without NO_VISION, or accept text-only service.

The engine refuses a metadata key while loading. The launcher uses the existing
binary by default. Rebuild it from the same tree with FORCE_BUILD=1; the base's
header uses the same deepseek4 keys as the V4.1 artifact, so an older binary is
the usual explanation.

Rollback

Stop this launcher, then start the older 1017 launcher again on 8008. Nothing in
this recipe modifies the older artifact or its directories; the only shared state
is the port and the single engine lock. The base and the fast-device files live
under new paths, so removing them later is a directory delete, not a surgery.
