#!/usr/bin/env bash
#
# pr_check.sh: verify the claims in the evals PR description against the repo.
#
# Usage (from anywhere inside the repo):
#     bash scripts/pr_check.sh
#     SKIP_CARGO=1 bash scripts/pr_check.sh       # static checks only
#     RUN_TIMEOUT=180 bash scripts/pr_check.sh    # seconds allowed for the router run
#     TARGET_BRANCHES="master master_fix" bash scripts/pr_check.sh
#
# Source checks read the committed tree (HEAD), so an uncommitted fix cannot pass.
# Cargo, eval, and router steps run against the working tree; a dirty tree is reported.
# Output worth pasting into the PR "Validation" section is saved under $OUT.

set -u
cd "$(git rev-parse --show-toplevel)" || exit 2

OUT="${OUT:-${TMPDIR:-/tmp}/pr_check}"
RUN_TIMEOUT="${RUN_TIMEOUT:-120}"
TARGET_BRANCHES="${TARGET_BRANCHES:-master master_fix}"
BIN="target/debug/cascade-router"
CAPTURE="logs/routing_capture.jsonl"
LEGACY_MODEL_PREFIX="c78a06de88"
mkdir -p "$OUT"

PASS=0
FAIL=0
WARN=0

pass()    { PASS=$((PASS + 1)); printf '  [PASS] %s\n' "$1"; }
fail()    { FAIL=$((FAIL + 1)); printf '  [FAIL] %s\n' "$1"; }
warn()    { WARN=$((WARN + 1)); printf '  [WARN] %s\n' "$1"; }
section() { printf '\n== %s ==\n' "$1"; }
indent()  { sed 's/^/        /'; }

# Contents of a file as committed at HEAD (empty if missing).
committed() { git show "HEAD:$1" 2>/dev/null; }

# has <file> <ERE> <description>: pattern must appear in the committed file.
has() {
    if committed "$1" | grep -Eq -- "$2"; then pass "$3"; else fail "$3"; fi
}

# lacks <file> <ERE> <description>: file must exist and the pattern must be absent.
lacks() {
    if [ -z "$(committed "$1")" ]; then
        fail "$3 ($1 missing at HEAD)"
    elif committed "$1" | grep -Eq -- "$2"; then
        fail "$3"
    else
        pass "$3"
    fi
}

# ---------------------------------------------------------------------------
section "Working tree"
if [ -z "$(git status --porcelain)" ]; then
    pass "working tree clean"
else
    warn "uncommitted changes: static checks read HEAD, cargo steps read the working tree"
    git status --short | indent
fi

# ---------------------------------------------------------------------------
section "Repo hygiene"
has .gitignore '^logs/?$' "logs/ is gitignored"
if [ -z "$(git ls-files logs)" ]; then pass "no files under logs/ are tracked"; else fail "tracked files under logs/"; git ls-files logs | indent; fi
has rust-toolchain.toml 'channel *= *"stable"' "rust-toolchain pins stable"
has rust-toolchain.toml 'rustfmt' "rust-toolchain includes rustfmt"
has rust-toolchain.toml 'clippy' "rust-toolchain includes clippy"

# ---------------------------------------------------------------------------
section "Cargo.toml"
for dep in serde_json chrono uuid sha2; do
    has Cargo.toml "^${dep} *=" "dependency: ${dep}"
done
if committed Cargo.toml | awk '
    /^\[profile\.dev\.package\.sha2\]/ { f = 1; next }
    /^\[/ { f = 0 }
    f && /opt-level *= *3/ { ok = 1 }
    END { exit !ok }'; then
    pass "[profile.dev.package.sha2] opt-level = 3"
else
    fail "[profile.dev.package.sha2] opt-level = 3 (startup hashing stalls in debug builds)"
fi

# ---------------------------------------------------------------------------
section "src/main.rs"
has src/main.rs 'fmt::layer\(' "stdout fmt layer is in the tracing registry"
has src/main.rs 'EnvFilter' "RUST_LOG goes through EnvFilter"
has src/main.rs '"info"' "default log level is info"
has src/main.rs 'sha256_file' "model and config hashes are computed at startup"
has src/main.rs 'tier2_small\.bin' "prefers models/tier2_small.bin"

# ---------------------------------------------------------------------------
section "src/router/cascade.rs"
sig=$(committed src/router/cascade.rs | perl -0777 -ne 'print $1 if /(pub\s+(?:async\s+)?fn\s+route\s*\(.*?\)\s*->\s*[^{]*)/s')
if [ -z "$sig" ]; then
    fail "could not find the route() signature"
else
    if printf '%s' "$sig" | grep -q 'RouteRecord'; then pass "route() returns a RouteRecord"; else fail "route() does not return a RouteRecord"; printf '%s\n' "$sig" | indent; fi
    if printf '%s' "$sig" | grep -q 'Hashes'; then pass "route() takes the file hashes"; else fail "route() does not take Hashes"; printf '%s\n' "$sig" | indent; fi
fi
has src/router/cascade.rs 'Uuid::new_v4\(\)' "request UUID is generated in cascade.rs"
has src/router/cascade.rs '\[request_id=' "log lines carry [request_id=...]"
lacks src/router/cascade.rs '\[req \{req\}\]' "no leftover [req N] log lines"

# ---------------------------------------------------------------------------
section "Tier 1 and Tier 2 source"
has src/router/tier1_runner.rs 'fn truncate_utf8' "UTF-8 safe truncation helper exists"
lacks src/router/tier1_runner.rs '&text\[\.\.max_bytes\]' "no raw byte slice at the imperative zone"
has src/router/tier1_runner.rs 'fn is_pure_arithmetic' "whole-prompt arithmetic check exists"
lacks src/router/tier1_runner.rs 'fn is_simple_arithmetic_prompt' "loose arithmetic check is gone"
has src/router/tier1_runner.rs 'truncate_utf8' "truncate_utf8 is referenced (tests or call site)"
lacks src/router/tier2_runner.rs 'fn arithmetic_jev_fastpath' "Tier 2 arithmetic fastpath is gone"
has src/router/tier2_runner.rs 'p_jev' "Tier 2 outcomes expose p_jev"
has src/router/tier2_runner.rs 'raw_probabilities' "Tier 2 outcomes expose raw_probabilities"
has src/router/tier2_runner.rs 'DEFAULT_CONFIDENCE_THRESHOLD' "DEFAULT_CONFIDENCE_THRESHOLD is exported"
lacks config/tier1.toml '"(convert |weather in |weather like|convert|weather)"' "convert/weather patterns removed from tier1.toml"

# ---------------------------------------------------------------------------
section "Evals, training script, README"
for f in evals/tier1/cases.jsonl examples/eval_tier1.rs examples/eval_tier2.rs calibration/make_fasttext_train.py; do
    if git cat-file -e "HEAD:$f" 2>/dev/null; then pass "committed: $f"; else fail "missing at HEAD: $f"; fi
done
if [ -f calibration/ft_valid.txt ]; then pass "calibration/ft_valid.txt exists"; else warn "calibration/ft_valid.txt missing (eval_tier2 reads it; run the training script)"; fi
if [ -f calibration/ft_train.txt ]; then pass "calibration/ft_train.txt exists"; else warn "calibration/ft_train.txt missing"; fi
rows=$(committed evals/tier1/cases.jsonl | grep -c .)
bad_labels=$(committed evals/tier1/cases.jsonl | grep -c . >/dev/null; committed evals/tier1/cases.jsonl | grep . | grep -Evc '"label": *"(jev_capable|needs_llm)"')
printf '        cases.jsonl rows: %s\n' "$rows"
if [ "$bad_labels" = "0" ]; then pass "every case has a valid label"; else fail "$bad_labels case(s) with a missing or unknown label"; fi
has calibration/make_fasttext_train.py '\-\-holdout-source' "training script has --holdout-source"
has calibration/make_fasttext_train.py '\-\-output-dir' "training script has --output-dir"
has README.md 'eval_tier1' "README documents eval_tier1"
has README.md 'eval_tier2' "README documents eval_tier2"
has README.md 'Request Capture' "README documents request capture"
if committed calibration/make_fasttext_train.py | grep -Eq 'MODEL_PATH *= *Path\("models/tier2\.bin"\)'; then
    warn "training script still writes models/tier2.bin (main prefers models/tier2_small.bin)"
fi

# ---------------------------------------------------------------------------
if [ "${SKIP_CARGO:-0}" = "1" ]; then
    section "Cargo, evals, router run"
    warn "skipped (SKIP_CARGO=1)"
else
    section "Build and tests"
    if cargo build --all-targets >"$OUT/build.log" 2>&1; then
        pass "cargo build --all-targets"
    else
        fail "cargo build --all-targets (log: $OUT/build.log)"
        tail -20 "$OUT/build.log" | indent
    fi
    if cargo test --all-targets >"$OUT/test.log" 2>&1; then
        totals=$(awk '/^test result:/ { p += $4; f += $6 } END { printf "%d passed, %d failed", p, f }' "$OUT/test.log")
        pass "cargo test --all-targets ($totals)"
    else
        fail "cargo test --all-targets (log: $OUT/test.log)"
        grep -E '^test .* FAILED|panicked' "$OUT/test.log" | head -10 | indent
    fi

    section "Tier 1 eval"
    if cargo run -q --example eval_tier1 >"$OUT/eval_tier1.txt" 2>&1; then
        pass "eval_tier1 ran"
    else
        fail "eval_tier1 failed (log: $OUT/eval_tier1.txt)"
        tail -15 "$OUT/eval_tier1.txt" | indent
    fi
    if grep -q '^\[FALSE JEV\]' "$OUT/eval_tier1.txt"; then
        fail "eval_tier1 reports false Jev commits"
        grep '^\[FALSE JEV\]' "$OUT/eval_tier1.txt" | indent
    else
        pass "no false Jev commits in eval_tier1"
    fi
    grep -E 'precision|coverage|decided rate' "$OUT/eval_tier1.txt" | indent

    section "Tier 2 eval"
    if cargo run -q --example eval_tier2 >"$OUT/eval_tier2.txt" 2>&1; then
        pass "eval_tier2 ran"
    else
        fail "eval_tier2 failed (log: $OUT/eval_tier2.txt)"
        tail -15 "$OUT/eval_tier2.txt" | indent
    fi
    tail -25 "$OUT/eval_tier2.txt" | indent

    section "Router run and request capture"
    if [ ! -x "$BIN" ]; then
        fail "binary not found at $BIN"
    else
        before=0
        [ -f "$CAPTURE" ] && before=$(wc -l <"$CAPTURE" | tr -d ' ')
        perl -e 'alarm shift; exec @ARGV' "$RUN_TIMEOUT" "$BIN" >"$OUT/run.log" 2>&1
        rc=$?
        if [ $rc -eq 0 ]; then
            pass "router exited cleanly"
        elif [ $rc -eq 142 ]; then
            fail "router did not exit within ${RUN_TIMEOUT}s (log: $OUT/run.log)"
        else
            fail "router exited with status $rc (log: $OUT/run.log)"
            tail -15 "$OUT/run.log" | indent
        fi

        after=0
        [ -f "$CAPTURE" ] && after=$(wc -l <"$CAPTURE" | tr -d ' ')
        delta=$((after - before))
        done_lines=$(grep -c ' DONE ' "$OUT/run.log")
        printf '        capture records added: %s, DONE log lines: %s\n' "$delta" "$done_lines"
        if [ "$delta" -ge 1 ]; then pass "capture file gained records"; else fail "capture file gained no records"; fi
        if [ "$delta" -eq "$done_lines" ]; then pass "one capture record per routed request"; else fail "capture records ($delta) != DONE lines ($done_lines)"; fi

        if grep -q 'decided_by=tier1' "$OUT/run.log"; then pass "a sample prompt resolved at tier 1"; else warn "no sample prompt resolved at tier 1"; fi
        if grep -q 'request_id=' "$OUT/run.log"; then pass "log lines carry request_id"; else fail "log lines carry no request_id"; fi

        if [ "$delta" -ge 1 ]; then
            while IFS=$'\t' read -r status msg; do
                case "$status" in
                    PASS) pass "$msg" ;;
                    FAIL) fail "$msg" ;;
                    WARN) warn "$msg" ;;
                esac
            done < <(python3 - "$CAPTURE" "$delta" "$OUT/run.log" "$LEGACY_MODEL_PREFIX" <<'PY'
import json, re, sys

path, n, log, legacy = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
lines = open(path, encoding="utf-8").read().splitlines()[-n:]
logtext = open(log, encoding="utf-8", errors="replace").read()

keys, values, ids, bad = set(), {}, [], 0

def walk(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            keys.add(k)
            if isinstance(v, str):
                values.setdefault(k, v)
            walk(v)
    elif isinstance(obj, list):
        for v in obj:
            walk(v)

for line in lines:
    try:
        rec = json.loads(line)
    except Exception:
        bad += 1
        continue
    walk(rec)
    ids.append(str(rec.get("request_id", "")))

def out(status, msg):
    print(f"{status}\t{msg}")

out("FAIL" if bad else "PASS", f"{bad} unparseable capture line(s)" if bad else "new capture records are valid JSON")

required = ["request_id", "prompt", "sample_prob", "tier1_toml", "tier2_bin"]
missing = [k for k in required if k not in keys]
out("FAIL" if missing else "PASS",
    f"capture records missing keys: {', '.join(missing)}" if missing else "capture records carry all required keys")

uuid_v4 = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")
bad_ids = [i for i in ids if not uuid_v4.match(i)]
out("FAIL" if bad_ids else "PASS", "request_id is not a UUID v4" if bad_ids else "request_id is a UUID v4 on every record")

unjoined = [i for i in ids if i and i not in logtext]
out("FAIL" if unjoined else "PASS",
    f"{len(unjoined)} capture request_id(s) absent from the router log" if unjoined
    else "every capture request_id appears in the router log")

out("PASS" if "p_jev" in keys else "WARN", "Tier 2 p_jev is recorded" if "p_jev" in keys else "no p_jev in new records (no prompt reached tier 2?)")
out("PASS" if "reason" in keys else "WARN", "outcomes carry a reason" if "reason" in keys else "no reason field in new records")

model_hash = values.get("tier2_bin", "")
if model_hash.startswith(legacy):
    out("WARN", f"tier2_bin hash {model_hash[:10]} is the legacy 800 MB model")
elif model_hash:
    out("PASS", f"tier2_bin hash {model_hash[:10]} is not the legacy 800 MB model")
PY
)
        fi
    fi

    # Paste-ready Validation section.
    {
        printf '### Tier 1 eval\n\n```\n'
        cat "$OUT/eval_tier1.txt"
        printf '```\n\n### Tier 2 eval\n\n```\n'
        cat "$OUT/eval_tier2.txt"
        printf '```\n'
    } >"$OUT/validation.md"
fi

# ---------------------------------------------------------------------------
section "Branch and merge"
cur=$(git rev-parse --abbrev-ref HEAD)
printf '  current branch: %s\n' "$cur"
first=1
for b in $TARGET_BRANCHES; do
    if ! git rev-parse --verify -q "$b" >/dev/null; then
        warn "branch $b not found locally"
        continue
    fi
    counts=$(git rev-list --left-right --count "$b...HEAD")
    behind=${counts%%[[:space:]]*}
    ahead=${counts##*[[:space:]]}
    printf '  vs %s: %s commit(s) behind, %s ahead (merge-base %s)\n' \
        "$b" "$behind" "$ahead" "$(git rev-parse --short "$(git merge-base "$b" HEAD)")"
    mt=$(git merge-tree --write-tree --name-only "$b" HEAD 2>&1)
    rc=$?
    if [ $rc -eq 0 ]; then
        pass "merges cleanly with $b"
    elif [ $rc -eq 1 ]; then
        files=$(printf '%s\n' "$mt" | sed -n '2,/^$/p' | sed '/^$/d' | tr '\n' ' ')
        if [ $first -eq 1 ]; then fail "merge with $b conflicts in: $files"; else warn "merge with $b conflicts in: $files"; fi
    else
        warn "git merge-tree --write-tree failed for $b (needs git 2.38+)"
    fi
    first=0
done

# ---------------------------------------------------------------------------
section "Summary"
printf '  %s passed, %s failed, %s warnings\n' "$PASS" "$FAIL" "$WARN"
[ -f "$OUT/validation.md" ] && printf '  paste-ready Validation section: %s\n' "$OUT/validation.md"
printf '  logs: %s\n' "$OUT"
[ "$FAIL" -eq 0 ]
