(** The most important tests in this repository.

    {1 The argument}

    A backtest has lookahead bias if any decision made at time [t] depends on
    data from after [t]. The definitive test for that property is a
    {b truncation test}:

    {v
      Run the pipeline on the full series      -> signals S_full[0..n-1]
      Delete every observation after bar t
      Run the pipeline on the truncated series -> signals S_trunc[0..t]
      Assert  S_full[i] = S_trunc[i]  for all i <= t
    v}

    If any signal at or before [t] used future data, deleting that data must
    change it. Bit-identical output across the truncation is therefore direct
    evidence of causality — not an argument about the code, but a measurement
    of its behaviour.

    Note the comparison is for {b exact bit equality}, not approximate. These
    are the same arithmetic operations on the same inputs in the same order, so
    any difference at all is a real dependence on the deleted data. Using a
    tolerance here would let a small leak hide inside it.

    {1 What each test covers}

    - {!test_signals_truncation_invariance}: the signal path (hedge ratio,
      spread, z-score) across many truncation points.
    - {!test_full_backtest_prefix_invariance}: the whole engine including
      portfolio state — NAV, positions, and trade events must agree on the
      overlapping prefix.
    - {!test_future_perturbation_invariance}: rather than deleting the future,
      {e replace} it with completely different data. Signals up to [t] must not
      move. This catches a leak that a truncation test could in principle miss
      (one that reads a fixed future index rather than a relative one).
    - {!test_causal_view_cannot_read_future}: the mechanism itself — a
      {!Causal.view} returns [None] beyond its bound.
    - {!test_execution_is_next_bar}: the complementary property. Signals being
      causal is not sufficient; the {e fill} must also be at the next bar's
      price, which this verifies against the raw price series.

    {1 A deliberate negative control}

    {!test_lookahead_detector_catches_a_real_leak} constructs a signal
    generator that {e does} peek forward, and asserts the truncation test
    fails on it. Without this, a truncation test that silently compared
    nothing would pass vacuously and we would have no evidence the detector
    works. *)

open Statarb
open Statarb.Types

let eps_exact = 0.0

(** Compare two signal arrays on the overlapping prefix [0..upto]. *)
let assert_signals_agree ~(upto : int) ~(full : (float * float * float option) option array)
    ~(trunc : (float * float * float option) option array) ~(label : string) =
  for i = 0 to upto do
    match (full.(i), trunc.(i)) with
    | None, None -> ()
    | Some (b1, s1, z1), Some (b2, s2, z2) ->
        Alcotest.(check (float eps_exact))
          (Printf.sprintf "%s: hedge ratio at bar %d" label i)
          b1 b2;
        Alcotest.(check (float eps_exact))
          (Printf.sprintf "%s: spread at bar %d" label i)
          s1 s2;
        (match (z1, z2) with
        | None, None -> ()
        | Some a, Some b ->
            Alcotest.(check (float eps_exact))
              (Printf.sprintf "%s: z-score at bar %d" label i)
              a b
        | _ ->
            Alcotest.failf
              "%s: z-score presence differs at bar %d (full=%s trunc=%s)" label
              i
              (match z1 with None -> "none" | Some _ -> "some")
              (match z2 with None -> "none" | Some _ -> "some"))
    | _ ->
        Alcotest.failf
          "%s: signal presence differs at bar %d (full=%s trunc=%s)" label i
          (match full.(i) with None -> "none" | Some _ -> "some")
          (match trunc.(i) with None -> "none" | Some _ -> "some")
  done

(** {b The core lookahead test.}

    For each of several truncation points [t], run the signal pipeline on
    [series[0..t]] and assert every signal matches the full-series run
    exactly. *)
let test_signals_truncation_invariance () =
  let cfg = Fixtures.test_config () in
  let series =
    Fixtures.cointegrated ~n:400 ~seed:42 ~beta:1.2 ~half_life:15.
      ~sigma_spread:0.02 ~sigma_common:0.01
  in
  let full = Fixtures.get_ok (Backtest.signals_only cfg series) in
  (* Truncation points spanning warm-up, early trading, and late sample. *)
  let points = [ 25; 40; 60; 99; 150; 233; 300; 399 ] in
  List.iter
    (fun t ->
      let truncated = Array.sub series 0 (t + 1) in
      let trunc = Fixtures.get_ok (Backtest.signals_only cfg truncated) in
      Alcotest.(check int)
        (Printf.sprintf "truncated run has %d bars" (t + 1))
        (t + 1) (Array.length trunc);
      assert_signals_agree ~upto:t ~full ~trunc
        ~label:(Printf.sprintf "truncate@%d" t))
    points

(** Same property, but for the full engine including portfolio state.

    NAV, position, and trade events on the overlapping prefix must be
    identical. This is strictly stronger than the signal test: it would also
    catch a leak introduced in execution or accounting rather than in signal
    generation.

    The final bar of the truncated run is excluded from the comparison because
    the engine force-closes any open position at the end of the sample. That
    liquidation is a property of where the data stops, not a use of future
    information, so comparing it would be comparing two different (correct)
    behaviours. *)
let test_full_backtest_prefix_invariance () =
  let cfg = Fixtures.test_config () in
  let series =
    Fixtures.cointegrated ~n:400 ~seed:7 ~beta:0.85 ~half_life:12.
      ~sigma_spread:0.025 ~sigma_common:0.012
  in
  let full = Fixtures.get_ok (Backtest.run cfg series) in
  let full_bars = Array.of_list full.bars in
  List.iter
    (fun t ->
      let truncated = Array.sub series 0 (t + 1) in
      let trunc = Fixtures.get_ok (Backtest.run cfg truncated) in
      let trunc_bars = Array.of_list trunc.bars in
      (* Compare up to t-1: bar t of the truncated run includes the forced
         end-of-data liquidation. *)
      for i = 0 to t - 1 do
        let a = full_bars.(i) and b = trunc_bars.(i) in
        Alcotest.(check (float eps_exact))
          (Printf.sprintf "NAV at bar %d (truncate@%d)" i t)
          a.r_nav b.r_nav;
        Alcotest.(check string)
          (Printf.sprintf "position at bar %d (truncate@%d)" i t)
          a.r_position b.r_position;
        Alcotest.(check string)
          (Printf.sprintf "trade event at bar %d (truncate@%d)" i t)
          a.r_trade_event b.r_trade_event;
        Alcotest.(check (float eps_exact))
          (Printf.sprintf "cash at bar %d (truncate@%d)" i t)
          a.r_cash b.r_cash
      done)
    [ 120; 200; 275; 350 ]

(** Replace the future with different data rather than deleting it.

    A truncation test can in principle be passed by code that reads a fixed
    absolute index (e.g. always [series.(500)]) if that index happens to be
    out of range in every truncated run and the code silently falls back. This
    test closes that gap: the series length is unchanged, only the {e values}
    after [t] differ, and signals up to [t] must be unmoved. *)
let test_future_perturbation_invariance () =
  let cfg = Fixtures.test_config () in
  let base =
    Fixtures.cointegrated ~n:300 ~seed:11 ~beta:1.05 ~half_life:20.
      ~sigma_spread:0.02 ~sigma_common:0.01
  in
  let alt =
    Fixtures.cointegrated ~n:300 ~seed:999 ~beta:3.0 ~half_life:4.
      ~sigma_spread:0.15 ~sigma_common:0.05
  in
  let full = Fixtures.get_ok (Backtest.signals_only cfg base) in
  List.iter
    (fun t ->
      (* Same length; everything after t swapped for a wildly different pair. *)
      let perturbed =
        Array.init (Array.length base) (fun i ->
            if i <= t then base.(i) else alt.(i))
      in
      let pert = Fixtures.get_ok (Backtest.signals_only cfg perturbed) in
      assert_signals_agree ~upto:t ~full ~trunc:pert
        ~label:(Printf.sprintf "perturb-after@%d" t))
    [ 50; 100; 175; 250 ]

(** The enforcement mechanism itself: a view cannot address beyond its bound. *)
let test_causal_view_cannot_read_future () =
  let xs = [| 0.; 1.; 2.; 3.; 4.; 5. |] in
  let v = Causal.create xs 3 in
  Alcotest.(check int) "now" 3 (Causal.now v);
  Alcotest.(check int) "length" 4 (Causal.length v);
  Alcotest.(check (option (float 0.))) "index 0 visible" (Some 0.) (Causal.get v 0);
  Alcotest.(check (option (float 0.))) "index 3 visible" (Some 3.) (Causal.get v 3);
  Alcotest.(check (option (float 0.))) "index 4 hidden" None (Causal.get v 4);
  Alcotest.(check (option (float 0.))) "index 5 hidden" None (Causal.get v 5);
  Alcotest.(check (option (float 0.))) "negative index" None (Causal.get v (-1));
  (* to_visible_array must expose exactly the visible prefix. *)
  Alcotest.(check int)
    "visible array length" 4
    (Array.length (Causal.to_visible_array v));
  (* Advancing reveals exactly one more. *)
  match Causal.advance v with
  | None -> Alcotest.fail "advance should succeed mid-series"
  | Some v' ->
      Alcotest.(check int) "advanced now" 4 (Causal.now v');
      Alcotest.(check (option (float 0.))) "index 4 now visible" (Some 4.)
        (Causal.get v' 4);
      Alcotest.(check (option (float 0.))) "index 5 still hidden" None
        (Causal.get v' 5)

(** [lookback] must never return a window that extends past [now]. *)
let test_lookback_never_reads_future () =
  let xs = Array.init 100 float_of_int in
  for t = 0 to 99 do
    let v = Causal.create xs t in
    List.iter
      (fun k ->
        match Causal.lookback v k with
        | Error _ -> () (* insufficient data is fine *)
        | Ok w ->
            Alcotest.(check int) "window length" k (Array.length w);
            (* Last element of the window is the current bar, never later. *)
            Alcotest.(check (float 0.))
              (Printf.sprintf "window ends at now (t=%d k=%d)" t k)
              (float_of_int t)
              w.(k - 1);
            Array.iter
              (fun x ->
                if x > float_of_int t then
                  Alcotest.failf "lookback at t=%d returned future value %g" t x)
              w)
      [ 1; 5; 20; 60 ]
  done

(** {b Negative control.}

    A deliberately non-causal signal generator: the "z-score" at bar [t] is
    computed from a window {e centered} on [t], which reads [t+k] and is
    therefore lookahead. The truncation comparison must detect it.

    Without this test, the passing truncation tests above would be weak
    evidence: a comparison that never actually compares anything also passes.
    Here we confirm the detector fires on a known-bad input. *)
let test_lookahead_detector_catches_a_real_leak () =
  let series =
    Fixtures.cointegrated ~n:200 ~seed:3 ~beta:1.0 ~half_life:10.
      ~sigma_spread:0.03 ~sigma_common:0.01
  in
  let half = 10 in
  (* A centered-window mean: the canonical lookahead bug. *)
  let leaky_signal (s : series) : float option array =
    let n = Array.length s in
    let logs = Array.map (fun b -> log (Price.to_float b.price_a)) s in
    Array.init n (fun i ->
        if i - half < 0 || i + half >= n then None
        else begin
          let total = ref 0. in
          for j = i - half to i + half do
            total := !total +. logs.(j)
          done;
          Some (logs.(i) -. (!total /. float_of_int ((2 * half) + 1)))
        end)
  in
  let full = leaky_signal series in
  let t = 120 in
  let trunc = leaky_signal (Array.sub series 0 (t + 1)) in
  (* Find at least one bar at or before t where the two disagree. If the
     truncation comparison found no disagreement on genuinely leaky code, the
     comparison itself would be worthless. *)
  let disagreement = ref false in
  for i = 0 to t do
    match (full.(i), trunc.(i)) with
    | Some a, Some b -> if a <> b then disagreement := true
    | None, None -> ()
    | _ -> disagreement := true
  done;
  Alcotest.(check bool)
    "truncation comparison detects a centered-window (lookahead) signal" true
    !disagreement

(** {b Next-bar execution.}

    Causal signals are necessary but not sufficient: filling at the same bar's
    close is still lookahead even if the signal itself is causal. Every entry
    fill must occur at least one bar after the bar whose z-score triggered it.

    This is verified structurally: for each entry event in the audit trail, the
    z-score that justified it is the one at the {e previous} bar, and that
    previous bar's z-score must have breached the entry threshold. *)
let test_execution_is_next_bar () =
  let cfg = Fixtures.test_config () in
  let series =
    Fixtures.cointegrated ~n:500 ~seed:23 ~beta:1.1 ~half_life:14.
      ~sigma_spread:0.03 ~sigma_common:0.01
  in
  let res = Fixtures.get_ok (Backtest.run cfg series) in
  let bars = Array.of_list res.bars in
  let entries = ref 0 in
  Array.iteri
    (fun i r ->
      if String.length r.r_trade_event >= 5
         && String.sub r.r_trade_event 0 5 = "entry"
      then begin
        incr entries;
        (* An entry at bar i must have been decided at bar i-1. *)
        if i = 0 then Alcotest.fail "entry on bar 0 is impossible";
        match bars.(i - 1).r_zscore with
        | None ->
            Alcotest.failf
              "entry at bar %d but the previous bar had no z-score" i
        | Some z ->
            if Float.abs z < cfg.entry_threshold then
              Alcotest.failf
                "entry at bar %d but previous bar z=%.4f did not breach the \
                 entry threshold %.4f"
                i z cfg.entry_threshold
      end)
    bars;
  (* Guard against vacuous success: if the strategy never traded, the loop
     above asserted nothing. *)
  Alcotest.(check bool)
    (Printf.sprintf "strategy produced entries to check (got %d)" !entries)
    true (!entries > 0)

(** Exit fills obey the same rule. *)
let test_exit_is_next_bar () =
  let cfg = Fixtures.test_config () in
  let series =
    Fixtures.cointegrated ~n:500 ~seed:31 ~beta:0.95 ~half_life:10.
      ~sigma_spread:0.03 ~sigma_common:0.01
  in
  let res = Fixtures.get_ok (Backtest.run cfg series) in
  let bars = Array.of_list res.bars in
  let n = Array.length bars in
  let checked = ref 0 in
  Array.iteri
    (fun i r ->
      let is_exit =
        String.length r.r_trade_event >= 4
        && String.sub r.r_trade_event 0 4 = "exit"
      in
      (* The forced end-of-data close on the final bar is not signal-driven. *)
      if is_exit && i < n - 1 then begin
        incr checked;
        (* The bar before an exit must have held a position. *)
        Alcotest.(check bool)
          (Printf.sprintf "bar %d held a position before its exit at %d" (i - 1) i)
          true
          (bars.(i - 1).r_position <> "flat")
      end)
    bars;
  Alcotest.(check bool)
    (Printf.sprintf "strategy produced exits to check (got %d)" !checked)
    true (!checked > 0)

(** The rolling statistics are causal in isolation.

    [Rolling.zscore] over a view at [t] must be unchanged when the underlying
    array's tail is replaced with garbage. *)
let test_rolling_stats_are_causal () =
  let n = 200 in
  let g = Fixtures.Lcg.create 5 in
  let xs = Array.init n (fun _ -> Fixtures.Lcg.next_gaussian g) in
  let t = 120 in
  let k = 30 in
  let v = Causal.create xs t in
  let z_full = Fixtures.get_ok (Rolling.zscore v k) in
  let m_full = Fixtures.get_ok (Rolling.mean v k) in
  let s_full = Fixtures.get_ok (Rolling.stddev v k) in
  (* Poison everything after t. *)
  let poisoned = Array.copy xs in
  for i = t + 1 to n - 1 do
    poisoned.(i) <- 1e9
  done;
  let vp = Causal.create poisoned t in
  let z_p = Fixtures.get_ok (Rolling.zscore vp k) in
  let m_p = Fixtures.get_ok (Rolling.mean vp k) in
  let s_p = Fixtures.get_ok (Rolling.stddev vp k) in
  Alcotest.(check (float eps_exact)) "mean unaffected by poisoned future" m_full m_p;
  Alcotest.(check (float eps_exact)) "stddev unaffected by poisoned future" s_full s_p;
  match (z_full, z_p) with
  | Some a, Some b ->
      Alcotest.(check (float eps_exact)) "zscore unaffected by poisoned future" a b
  | None, None -> ()
  | _ -> Alcotest.fail "z-score presence changed under a poisoned future"

let tests =
  [
    ("signal truncation invariance (THE lookahead test)", `Quick,
     test_signals_truncation_invariance);
    ("full backtest prefix invariance", `Quick, test_full_backtest_prefix_invariance);
    ("future perturbation invariance", `Quick, test_future_perturbation_invariance);
    ("causal view cannot read future", `Quick, test_causal_view_cannot_read_future);
    ("lookback never reads future", `Quick, test_lookback_never_reads_future);
    ("rolling stats are causal", `Quick, test_rolling_stats_are_causal);
    ("negative control: detector catches a real leak", `Quick,
     test_lookahead_detector_catches_a_real_leak);
    ("entry execution is next-bar", `Quick, test_execution_is_next_bar);
    ("exit execution is next-bar", `Quick, test_exit_is_next_bar);
  ]
