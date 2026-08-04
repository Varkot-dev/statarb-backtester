(** Tests for the leakage-calibration instrument.

    This module is a deliberately corrupted signal generator, so its tests have
    an unusual job: they must confirm the corruption is {e real and controlled}
    — that dose zero is genuinely honest, that a non-zero dose genuinely
    cheats, and that each leak type moves performance in the direction the
    README claims.

    Without these, the dose-response curves in the README could be measuring
    anything. *)

open Statarb
open Statarb.Types

(** {b The anchor test.}

    At zero peek, the leaky path must produce exactly the same z-scores as the
    honest engine. If it did not, the calibration curve's origin would be
    somewhere other than the real backtest, and every comparison drawn from it
    would be against the wrong baseline.

    This is also a second, independent check on the production engine: two
    separately written implementations of the same causal calculation agreeing
    bit-for-bit is stronger evidence than either one alone. *)
let test_zero_peek_matches_the_honest_engine () =
  let cfg = Fixtures.test_config () in
  let series =
    Fixtures.cointegrated ~n:400 ~seed:1234 ~beta:1.1 ~half_life:14.
      ~sigma_spread:0.03 ~sigma_common:0.012
  in
  let honest = Fixtures.get_ok (Backtest.signals_only cfg series) in
  let spreads = Fixtures.get_ok (Leakage.spread_series cfg series) in
  let leaky = Leakage.leaky_zscores cfg spreads ~peek_bars:0 () in

  let compared = ref 0 in
  Array.iteri
    (fun i h ->
      match (h, leaky.(i)) with
      | Some (_, _, Some hz), Some lz ->
          incr compared;
          Alcotest.(check (float 1e-12))
            (Printf.sprintf "z-score at bar %d" i)
            hz lz
      | Some (_, _, None), None -> ()
      | None, None -> ()
      | _ ->
          Alcotest.failf "z-score presence differs at bar %d (honest=%s leaky=%s)"
            i
            (match h with Some (_, _, Some _) -> "some" | _ -> "none")
            (match leaky.(i) with Some _ -> "some" | None -> "none"))
    honest;
  Alcotest.(check bool)
    (Printf.sprintf "compared a meaningful number of bars (got %d)" !compared)
    true (!compared > 200)

(** {b The corruption is real.}

    With [k > 0], z-scores must actually differ from the honest ones. A
    calibration instrument that silently did nothing would produce a flat curve
    and a false sense of security. *)
let test_nonzero_peek_actually_differs () =
  let cfg = Fixtures.test_config () in
  let series =
    Fixtures.cointegrated ~n:400 ~seed:99 ~beta:1.0 ~half_life:12.
      ~sigma_spread:0.03 ~sigma_common:0.012
  in
  let spreads = Fixtures.get_ok (Leakage.spread_series cfg series) in
  let honest = Leakage.leaky_zscores cfg spreads ~peek_bars:0 () in
  List.iter
    (fun k ->
      let leaky = Leakage.leaky_zscores cfg spreads ~peek_bars:k () in
      let differences = ref 0 in
      Array.iteri
        (fun i h ->
          match (h, leaky.(i)) with
          | Some a, Some b -> if Float.abs (a -. b) > 1e-12 then incr differences
          | _ -> ())
        honest;
      Alcotest.(check bool)
        (Printf.sprintf "peek=%d changes many z-scores (changed %d)" k
           !differences)
        true (!differences > 100))
    [ 1; 3; 10 ]

(** {b The corrupted path fails the truncation test.}

    This is the sharpest statement the repository can make about its own
    lookahead suite. The honest engine passes truncation invariance; this
    module, given [k > 0], must {e fail} it. That confirms the test in
    [test_lookahead.ml] is sensitive to exactly the bias it claims to detect,
    at exactly the magnitudes the calibration curve reports.

    A truncation test that passed on both would be worthless. *)
let test_leaky_signals_break_truncation_invariance () =
  let cfg = Fixtures.test_config () in
  let series =
    Fixtures.cointegrated ~n:400 ~seed:77 ~beta:1.05 ~half_life:13.
      ~sigma_spread:0.03 ~sigma_common:0.012
  in
  let t = 250 in
  let truncated = Array.sub series 0 (t + 1) in

  let full_spreads = Fixtures.get_ok (Leakage.spread_series cfg series) in
  let trunc_spreads = Fixtures.get_ok (Leakage.spread_series cfg truncated) in

  (* At k = 0 the prefix must agree — the honest baseline. *)
  let full_0 = Leakage.leaky_zscores cfg full_spreads ~peek_bars:0 () in
  let trunc_0 = Leakage.leaky_zscores cfg trunc_spreads ~peek_bars:0 () in
  for i = 0 to t do
    match (full_0.(i), trunc_0.(i)) with
    | Some a, Some b ->
        Alcotest.(check (float 1e-12))
          (Printf.sprintf "k=0 is causal at bar %d" i)
          a b
    | None, None -> ()
    | _ -> Alcotest.failf "k=0 presence differs at bar %d" i
  done;

  (* At k = 5 the prefix must DISAGREE near the truncation point, because those
     bars were standardised against data that no longer exists. *)
  let full_5 = Leakage.leaky_zscores cfg full_spreads ~peek_bars:5 () in
  let trunc_5 = Leakage.leaky_zscores cfg trunc_spreads ~peek_bars:5 () in
  let disagreements = ref 0 in
  for i = 0 to t do
    match (full_5.(i), trunc_5.(i)) with
    | Some a, Some b -> if Float.abs (a -. b) > 1e-12 then incr disagreements
    | None, None -> ()
    | _ -> incr disagreements
  done;
  Alcotest.(check bool)
    (Printf.sprintf
       "k=5 breaks truncation invariance as it must (%d disagreements)"
       !disagreements)
    true (!disagreements > 0)

(** {b The dose-response curve is well-formed.}

    Every run must complete and produce finite, well-signed metrics. This
    deliberately asserts no direction: the measured timing-shift curve is
    non-monotone (roughly flat for k <= 2, then falling sharply), and pinning a
    shape here would encode today's dataset rather than a property of the
    instrument. Direction is asserted separately, per leak type, below. *)
let test_sweep_produces_a_wellformed_curve () =
  let cfg = Fixtures.test_config () in
  let series =
    Fixtures.cointegrated ~n:500 ~seed:31 ~beta:1.15 ~half_life:15.
      ~sigma_spread:0.03 ~sigma_common:0.012
  in
  let curve = Fixtures.get_ok (Leakage.sweep cfg series ~max_peek:8) in
  Alcotest.(check int) "one row per peek level" 9 (List.length curve);
  List.iteri
    (fun i (c : Leakage.calibration) ->
      Alcotest.(check int) "rows are in peek order" i c.peek_bars;
      Alcotest.(check bool)
        (Printf.sprintf "peek=%d Sharpe is finite" c.peek_bars)
        true
        (Float.is_finite c.sharpe);
      Alcotest.(check bool)
        (Printf.sprintf "peek=%d drawdown is non-positive" c.peek_bars)
        true
        (c.max_drawdown <= 0.);
      Alcotest.(check bool)
        (Printf.sprintf "peek=%d NAV is positive" c.peek_bars)
        true
        (c.final_nav > 0.))
    curve

(** {b Timing-shift leakage {e degrades} a mean-reversion strategy.}

    This test asserts the direction that was actually measured, which is the
    opposite of the intuition it replaced. The first version of this test
    asserted that foresight would inflate Sharpe; it failed, and investigating
    why produced the finding the README now reports.

    The reason is specific to the entry rule. The strategy wants to enter {e at}
    an extreme. A signal describing bar [t + k] makes it enter early, so it
    carries the position while the spread travels {e into} the extreme and only
    then collects the reversion.

    Averaged across seeds: the claim is about the systematic effect, not about
    one sample path. *)
let test_timing_shift_degrades_performance () =
  let cfg = Fixtures.test_config () in
  let honest_total = ref 0. and shifted_total = ref 0. in
  let n_seeds = 12 in
  for seed = 1 to n_seeds do
    let series =
      Fixtures.cointegrated ~n:400 ~seed ~beta:1.1 ~half_life:14.
        ~sigma_spread:0.03 ~sigma_common:0.012
    in
    let h = Fixtures.get_ok (Leakage.run_with_leakage cfg series ~peek_bars:0 ()) in
    let s = Fixtures.get_ok (Leakage.run_with_leakage cfg series ~peek_bars:5 ()) in
    honest_total := !honest_total +. h.sharpe;
    shifted_total := !shifted_total +. s.sharpe
  done;
  let honest_mean = !honest_total /. float_of_int n_seeds in
  let shifted_mean = !shifted_total /. float_of_int n_seeds in
  Alcotest.(check bool)
    (Printf.sprintf
       "5 bars of timing shift degrades mean Sharpe (honest %.4f -> shifted \
        %.4f)"
       honest_mean shifted_mean)
    true
    (shifted_mean < honest_mean)

(** {b Outcome-filter leakage {e inflates} the reported Sharpe.}

    The complement, and the leak that actually makes fraudulent backtests look
    good. Skipping trades that are going to lose must raise Sharpe, raise the
    win rate, and lower the trade count — the three-part fingerprint the README
    describes. *)
let test_outcome_filter_inflates_sharpe () =
  let cfg = Fixtures.test_config () in
  let series =
    Fixtures.cointegrated ~n:800 ~seed:404 ~beta:1.1 ~half_life:14.
      ~sigma_spread:0.03 ~sigma_common:0.012
  in
  let honest =
    Fixtures.get_ok
      (Leakage.run_with_outcome_filter cfg series ~fraction:0. ~seed:1)
  in
  let cheating =
    Fixtures.get_ok
      (Leakage.run_with_outcome_filter cfg series ~fraction:1.0 ~seed:1)
  in
  Alcotest.(check bool)
    (Printf.sprintf "skipping every loser raises Sharpe (%.4f -> %.4f)"
       honest.sharpe cheating.sharpe)
    true
    (cheating.sharpe > honest.sharpe);
  Alcotest.(check bool)
    (Printf.sprintf "and raises the win rate (%.3f -> %.3f)" honest.win_rate
       cheating.win_rate)
    true
    (cheating.win_rate > honest.win_rate);
  Alcotest.(check bool)
    (Printf.sprintf "and takes fewer trades (%d -> %d)" honest.n_trades
       cheating.n_trades)
    true
    (cheating.n_trades < honest.n_trades)

(** Dose zero of the outcome filter must be the honest engine exactly. *)
let test_outcome_filter_zero_dose_is_honest () =
  let cfg = Fixtures.test_config () in
  let series =
    Fixtures.cointegrated ~n:500 ~seed:55 ~beta:1.0 ~half_life:12.
      ~sigma_spread:0.03 ~sigma_common:0.012
  in
  let honest = Fixtures.get_ok (Backtest.run cfg series) in
  let filtered =
    Fixtures.get_ok
      (Leakage.run_with_outcome_filter cfg series ~fraction:0. ~seed:7)
  in
  Alcotest.(check (float 1e-12))
    "zero dose reproduces the honest Sharpe" honest.metrics.sharpe_ratio
    filtered.sharpe;
  Alcotest.(check int)
    "zero dose reproduces the honest trade count" honest.metrics.n_trades
    filtered.n_trades

(** A suppressed entry must skip the whole episode, not merely delay it.

    Declining one bar is not enough: the strategy would re-enter on the next bar
    still inside the entry band, so the losing trade would simply happen one bar
    later. Measured before the fix: skipping 100% of losers removed 2 of 54
    trades. This test pins the corrected behaviour. *)
let test_outcome_filter_skips_episodes_not_bars () =
  let cfg = Fixtures.test_config () in
  let series =
    Fixtures.cointegrated ~n:800 ~seed:404 ~beta:1.1 ~half_life:14.
      ~sigma_spread:0.03 ~sigma_common:0.012
  in
  let honest =
    Fixtures.get_ok
      (Leakage.run_with_outcome_filter cfg series ~fraction:0. ~seed:1)
  in
  let cheating =
    Fixtures.get_ok
      (Leakage.run_with_outcome_filter cfg series ~fraction:1.0 ~seed:1)
  in
  let removed = honest.n_trades - cheating.n_trades in
  let losers = honest.n_trades - int_of_float (Float.round (honest.win_rate *. float_of_int honest.n_trades)) in
  (* Skipping every loser should remove a substantial share of them, not a
     token handful. Not all: a skipped episode changes the subsequent path, so
     the counts do not have to match exactly. *)
  Alcotest.(check bool)
    (Printf.sprintf
       "skipping every loser removes a meaningful share of trades (removed %d \
        of ~%d losers)"
       removed losers)
    true
    (removed >= losers / 2)

let test_outcome_filter_rejects_a_bad_fraction () =
  let cfg = Fixtures.test_config () in
  let series =
    Fixtures.cointegrated ~n:200 ~seed:5 ~beta:1.0 ~half_life:10.
      ~sigma_spread:0.03 ~sigma_common:0.01
  in
  List.iter
    (fun f ->
      match Leakage.run_with_outcome_filter cfg series ~fraction:f ~seed:1 with
      | Error (Config_error _) -> ()
      | _ -> Alcotest.failf "fraction %g should be rejected" f)
    [ -0.1; 1.5 ]

let test_sweep_rejects_a_negative_peek () =
  let cfg = Fixtures.test_config () in
  let series =
    Fixtures.cointegrated ~n:200 ~seed:5 ~beta:1.0 ~half_life:10.
      ~sigma_spread:0.03 ~sigma_common:0.01
  in
  match Leakage.run_with_leakage cfg series ~peek_bars:(-1) () with
  | Error (Config_error _) -> ()
  | _ -> Alcotest.fail "a negative peek should be a Config_error"

let tests =
  [
    ("zero peek matches the honest engine (anchor)", `Quick,
     test_zero_peek_matches_the_honest_engine);
    ("non-zero peek actually differs", `Quick, test_nonzero_peek_actually_differs);
    ("leaky signals break truncation invariance", `Quick,
     test_leaky_signals_break_truncation_invariance);
    ("sweep produces a well-formed curve", `Quick,
     test_sweep_produces_a_wellformed_curve);
    ("timing shift DEGRADES performance (counterintuitive)", `Quick,
     test_timing_shift_degrades_performance);
    ("outcome filtering INFLATES Sharpe", `Quick,
     test_outcome_filter_inflates_sharpe);
    ("outcome filter at zero dose is the honest engine", `Quick,
     test_outcome_filter_zero_dose_is_honest);
    ("outcome filter skips episodes, not single bars", `Quick,
     test_outcome_filter_skips_episodes_not_bars);
    ("outcome filter rejects a bad fraction", `Quick,
     test_outcome_filter_rejects_a_bad_fraction);
    ("sweep rejects a negative peek", `Quick, test_sweep_rejects_a_negative_peek);
  ]
