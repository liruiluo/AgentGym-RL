# r112 update-1 token-credit gate

`agentmemorygym_verl.update1_token_credit_gate` is the run-owner gate for the
first completed optimizer update of the r112 SAO formal lineage. It is a
read-only observer until it publishes its own receipt. It does not modify the
learner, rollout, environment, reward, schedule, or lifecycle implementation.

## Invocation

Start the gate after the formal owner has published
`formal-owner/orchestrator-process-identity.json`. The output must be a
dedicated path inside the run directory and must not overlap a launch-bound
input or the formal-owner identity receipt.

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

Exit code `0` means that an atomic `PASS_CONTINUE` receipt was published and
the exact formal owner was left running. Exit code `1` means the receipt is a
failure. Missing or incomplete update-1 artifacts are polled until timeout;
complete contradictory evidence fails immediately.

`--dry-run` exercises the same evidence and owner-authentication path but
never signals a process, including on failure or timeout.

## Evidence contract

| Gate field | Run-owned evidence |
|---|---|
| run/source/config identity | `launch-receipt.json`, its bound source lock, resolved config, Hydra config, route registry, and schedule certificate |
| 64 complete episodes and all four routes | `rollout_data/1.jsonl`, joined to the frozen schedule |
| actor/critic gradients and actor K1 / critic K2 | distinct learner-owner row at FileLogger `step=1` |
| IcePop applied weights, ESS, and `high + low = OOB` | actor-owned `actor/rollout_corr/*` metrics at FileLogger `step=1` |
| publication version 1 | learner consumed version `0`, sync cadence `1`, and the distinct rollouter-owned `step=1` reset row with positive `version_time` |
| failure/cancel/overflow/stale-drop zero | learner and rollouter owner counters at FileLogger `step=1` |
| freeze set, optimizer membership, frozen/trainable deltas, critic LR | run-owned `critic-parameter-freeze.json`; LR must be `5e-7`, then `1e-6` |
| stop authority | bound formal-owner receipt plus live `/proc/<pid>/{stat,cmdline,environ}` |

The receipt includes these paths and observed SHA256/file bindings. The
launch receipt and freeze manifest are rebound before a PASS is published.

## Exact-stop semantics

On failure, the gate rereads the same owner receipt and requires unchanged
device/inode/ctime/size/SHA256. It then revalidates:

- schema and owner name;
- PID, start ticks, process group, and session (the owner must be its own
  group/session leader);
- the bootstrap path and exact command suffix;
- unique `--run-dir` and `--experiment-name` bindings;
- both `AMG_MULTITASK_RUN_ID` and `AGENTMEMORY_RUN_ID` in the live process
  environment.

Only then may it send `SIGTERM` through the existing pidfd-based exact-identity
helper. The process bootstrap remains responsible for its owned descendants.
If the owner receipt or live identity is missing, replaced, or inconsistent,
the receipt says `FAIL_NO_UNSAFE_STOP` and no process is signalled. There is no
name-based, numeric-group, or raw-PID fallback and no automatic escalation to
`SIGKILL`.

This gate closes the update-1 operational check only. It does not sign the
complete launch package, long-horizon training result, or external evaluation.
