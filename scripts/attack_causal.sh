#!/usr/bin/env bash
# Compile-fail attacks on the Causal abstraction boundary.
#
# WHY THIS EXISTS AS A SHELL SCRIPT RATHER THAN A UNIT TEST
#
# The property being asserted is "this program does NOT compile". A normal test
# cannot express that: anything in test/ must itself compile to run. So the
# whole 9-test lookahead suite validated the pipeline's *behaviour* under
# truncation while never touching the *mechanism's* integrity — and a
# reviewer's four-line attack program compiled and read the future straight
# through the public API.
#
# That is the general lesson: a test suite that only exercises the happy path
# of an abstraction cannot detect that the abstraction has no walls. Each case
# below is a real attack that DID work before engine/lib/causal.mli existed.
#
# Exit 0 iff every attack is rejected by the compiler.
set -uo pipefail
cd "$(dirname "$0")/../engine" || exit 1
eval "$(opam env 2>/dev/null)" || true

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/atk"
cat > "$WORK/atk/dune" <<'EOF'
(executable (name atk) (libraries statarb))
EOF

# Compile a snippet against the library. Returns 0 if it builds.
try_build() {
  printf 'open Statarb\nlet () =\n%s\n' "$1" > "$WORK/atk/atk.ml"
  rm -rf atk_tmp
  cp -r "$WORK/atk" atk_tmp
  local out
  out="$(dune build atk_tmp/atk.exe 2>&1)"
  local code=$?
  LAST_BUILD_OUTPUT="$out"
  rm -rf atk_tmp
  return "$code"
}

# POSITIVE CONTROL.
#
# Without this the harness is vacuous: `dune build` failing for ANY reason —
# a bad path, a stale _build, a typo in the generated file — would be read as
# "attack rejected", and the suite would pass green while testing nothing.
# That is exactly the failure this whole script exists to catch, and the first
# draft of it had precisely that bug. The control must compile; if it does not,
# the harness is broken and says so rather than reporting a false pass.
positive_control() {
  if try_build '  let v = Causal.create (Array.init 10 float_of_int) 2 in
  Printf.printf "%d\n" (Causal.now v)'; then
    echo "  ok    positive control — the harness can build a legal program"
    return 0
  fi
  echo "  BROKEN  positive control failed to compile."
  echo "          The harness cannot build ANY program, so its results are"
  echo "          meaningless. Build output:"
  echo "$LAST_BUILD_OUTPUT" | sed 's/^/            /' | head -15
  return 1
}

# An attack passes only when the compiler rejects it FOR THE RIGHT REASON.
# Requiring the error to mention the abstract type distinguishes "the boundary
# held" from "the build broke for an unrelated reason".
attack() {
  local name="$1" body="$2"
  if try_build "$body"; then
    echo "  FAIL  $name — compiled, so the future is reachable"
    return 1
  fi
  if echo "$LAST_BUILD_OUTPUT" | grep -qE "Unbound record field|Causal.data|Causal.now|is abstract"; then
    echo "  ok    $name — rejected at the abstraction boundary"
    return 0
  fi
  echo "  FAIL  $name — did not compile, but not because of the abstraction:"
  echo "$LAST_BUILD_OUTPUT" | sed 's/^/          /' | head -8
  return 1
}

echo "Attacking the Causal abstraction boundary:"
rc=0
positive_control || { echo; echo "Harness is broken; not reporting attack results."; exit 2; }
attack "read the underlying array field" \
'  let v = Causal.create (Array.init 10 float_of_int) 2 in
  ignore v.Causal.data' || rc=1

attack "forge a view with an arbitrary bound" \
'  ignore { Causal.data = Array.init 10 float_of_int; now = 9 }' || rc=1

attack "pattern-match the bound out of a view" \
'  let v = Causal.create (Array.init 10 float_of_int) 2 in
  let { Causal.data = d; now = _ } = v in
  ignore d' || rc=1

attack "mutate the bound in place" \
'  let v = Causal.create (Array.init 10 float_of_int) 2 in
  ignore { v with Causal.now = 9 }' || rc=1

echo
if [ "$rc" -eq 0 ]; then
  echo "All attacks rejected. The Causal abstraction holds."
else
  echo "AT LEAST ONE ATTACK COMPILED — the no-lookahead guarantee is not enforced."
fi
exit "$rc"
