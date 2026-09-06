# r112 update-1 token-credit gate

`agentmemorygym_verl.update1_token_credit_gate` is the evidence observer for the
first completed optimizer update of the r112 SAO formal lineage. It only reads
evidence and publishes its own receipt. It never signals a process and does not
modify the learner, rollout, environment, reward, schedule, or lifecycle.

## Invocation

Start the gate after the formal owner has published
`formal-owner/orchestrator-process-identity.json`. The only accepted output is
the fixed run-owned path `$RUN/gates/update1-token-credit.json`. A missing or
invalid launch receipt, or an incomplete protected-input set, aborts without
writing a gate receipt.

```bash
python -m agentmemorygym_verl.update1_token_credit_gate \
  --run-dir "$RUN" \
  --run-id "$RUN_ID" \
  --owner-receipt "$E/formal-owner/orchestrator-process-identity.json" \
  --expected-outer-commit "$OUTER_COMMIT" \
  --expected-inner-commit "$INNER_COMMIT" \
  --expected-verl-commit "$VERL_COMMIT" \
  --output "$RUN/gates/update1-token-credit.json" \
  --timeout-seconds 1800 \
  --poll-seconds 1
```

Exit code `0` means that an atomic `PASS_CONTINUE` receipt was published. That
decision means only that the frozen update-1 evidence contract passed; it does
not assert continuing owner authority or itself authorize process control.
Exit code `1` means that an atomic `FAIL_NO_UNSAFE_STOP` receipt was published.
The package-v2 runner consumes the receipt and remains the sole lifecycle
authority for its direct formal-supervisor child. Missing or incomplete
update-1 artifacts are polled until timeout; complete contradictory evidence
fails immediately.

The timeout, polling, and stop-timeout values must be finite real numbers.
Timeouts may be zero, while the polling interval must be positive.

`--dry-run`, `--owner-receipt`, and `--stop-timeout-seconds` remain CLI/API
compatible. Dry-run follows the same observer path. The stop timeout has no
process-control effect because this module never signals a process.

## Evidence contract

| Gate field | Run-owned evidence |
|---|---|
| run/source/config identity | `launch-receipt.json`, its bound source lock, resolved config, Hydra config, route registry, schedule certificate, and frozen schedule |
| 64 complete episodes and all four routes | `rollout_data/1.jsonl`, joined to the frozen schedule |
| actor/critic gradients and actor K1 / critic K2 | distinct learner-owner row at FileLogger `step=1` |
| IcePop applied weights, ESS, and `high + low = OOB` | actor-owned `actor/rollout_corr/*` metrics at FileLogger `step=1` |
| publication version 1 | learner consumed version `0`, sync cadence `1`, and the distinct rollouter-owned `step=1` reset row with positive `version_time` |
| failure/cancel/overflow/stale-drop zero | learner and rollouter owner counters at FileLogger `step=1` |
| freeze set, optimizer membership, frozen/trainable deltas, critic LR | run-owned `critic-parameter-freeze.json`; LR must be `5e-7`, then `1e-6` |
| owner provenance | bound formal-owner receipt plus a point-in-time `/proc/<pid>/{stat,cmdline,environ}` observation |

The receipt includes these paths and observed SHA256/file bindings. A PASS is
published only after a second complete update-1 audit, followed by a rebind of
every immutable launch/config/source/schedule input and the critic freeze
manifest. The gate also reselects and rehashes the complete rows in
`rollout_data/1.jsonl` and the FileLogger rows whose integer `step == 1`.
Later FileLogger steps may continue to append, but the bounded step-1 view may
not change. After that rebind, the owner receipt is bound once more for
publication-time provenance. This is not a sustained owner lease; the runner
must independently check that its directly owned child is still live before it
acts on the receipt.

## Observer / runner lifecycle boundary

The evidence observer never opens a pidfd, sends `SIGTERM`, invokes a callback,
or waits for process exit. Every PASS and FAIL receipt therefore records:

- `termination.requested = false`;
- `termination.signal = null`;
- `termination.scope = observer-only`;
- `termination.status = not-owned-by-observer`.

The package-v2 runner is the direct parent of the fallback formal supervisor.
It consumes only the fixed receipt's `status` field. On `fail`, it verifies its
own direct-child relationship, sends TERM only to that direct child, waits for
the child, and performs its normal cleanup. On `pass`, it continues waiting for
that child. Those runner guarantees are outside this observer module; the
owner receipt embedded here is read-only provenance and can never trigger a
signal.

## Descriptor-anchored publication

PASS and FAIL use the same publisher. It opens the run directory with
`O_DIRECTORY|O_NOFOLLOW`, creates/opens `gates` relative to that descriptor,
and retains both directory descriptors through publication. The temporary
receipt is created with `O_CREAT|O_EXCL|O_NOFOLLOW`, written and fsynced, then
renamed to the fixed basename with `src_dir_fd` and `dst_dir_fd`; the gate
directory is fsynced afterward. Immediately before and after replacement, the
named run/gates device and inode must still match the retained descriptors.
Replacing `$RUN/gates` with a symlink therefore fails closed and cannot redirect
either a PASS or FAIL receipt outside the run.

This gate closes the update-1 operational check only. It does not sign the
complete launch package, long-horizon training result, or external evaluation.
