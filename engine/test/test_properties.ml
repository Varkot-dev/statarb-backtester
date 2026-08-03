(** Property-based tests.

    Example-based tests confirm behaviour on the cases the author thought of.
    These assert invariants over randomly generated inputs, which is how you
    find the cases nobody thought of — a NAV path that is monotone except for
    one bar, a window where every value is identical, a return series that is
    all zeros.

    Each property is stated as a mathematical claim in its docstring; if the
    claim is false the generator will eventually produce a counterexample and
    QCheck will shrink it to a minimal one. *)

open Statarb

let n_runs = 500

(** Generator for a strictly positive NAV path. Values are bounded away from
    zero because a wiped-out account makes drawdown ratios undefined, which the
    implementation correctly refuses. *)
let positive_navs =
  QCheck.(
    map
      (fun xs -> Array.of_list (List.map (fun x -> 1. +. Float.abs x) xs))
      (list_size Gen.(int_range 2 200) (float_bound_inclusive 1e6)))

(** Generator for a finite return series. *)
let finite_returns =
  QCheck.(
    map
      (fun xs -> Array.of_list xs)
      (list_size Gen.(int_range 2 200) (float_range (-0.5) 0.5)))

(** {b Max drawdown is always <= 0.}

    By construction [nav_t / peak_t <= 1] since [peak_t >= nav_t], so the
    minimum of [nav/peak - 1] cannot exceed 0. *)
let prop_max_drawdown_is_non_positive =
  QCheck.Test.make ~count:n_runs ~name:"max drawdown is always <= 0"
    positive_navs (fun navs ->
      match Metrics.max_drawdown navs with
      | Error _ -> false
      | Ok (dd, _) -> dd <= 0.)

(** {b Max drawdown never exceeds a total loss.}

    NAV is strictly positive, so [nav/peak > 0] and the drawdown is strictly
    greater than -1. A backtester reporting a drawdown worse than -100% on a
    positive NAV path has a sign or ordering bug. *)
let prop_max_drawdown_above_total_loss =
  QCheck.Test.make ~count:n_runs
    ~name:"max drawdown is strictly greater than -1 (no worse than total loss)"
    positive_navs (fun navs ->
      match Metrics.max_drawdown navs with
      | Error _ -> false
      | Ok (dd, _) -> dd > -1.)

(** {b Max drawdown is scale-invariant.}

    Multiplying every NAV by a positive constant is a change of currency units;
    the drawdown is a ratio and must not move. *)
let prop_max_drawdown_scale_invariant =
  QCheck.Test.make ~count:n_runs ~name:"max drawdown is scale-invariant"
    QCheck.(pair positive_navs (float_range 0.01 1000.))
    (fun (navs, k) ->
      let scaled = Array.map (fun v -> v *. k) navs in
      match (Metrics.max_drawdown navs, Metrics.max_drawdown scaled) with
      | Ok (a, _), Ok (b, _) -> Float.abs (a -. b) < 1e-9
      | _ -> false)

(** {b A monotonically non-decreasing NAV path has zero drawdown.}

    Sorting a path ascending makes it monotone; the peak always equals the
    current value, so every drawdown is exactly 0. *)
let prop_monotone_path_has_no_drawdown =
  QCheck.Test.make ~count:n_runs
    ~name:"a monotonically increasing NAV path has zero drawdown" positive_navs
    (fun navs ->
      let sorted = Array.copy navs in
      Array.sort Float.compare sorted;
      match Metrics.max_drawdown sorted with
      | Ok (dd, _) -> Float.abs dd < 1e-12
      | Error _ -> false)

(** {b Drawdown duration is non-negative and within the sample.} *)
let prop_drawdown_duration_in_range =
  QCheck.Test.make ~count:n_runs
    ~name:"drawdown duration is within [0, n-1]" positive_navs (fun navs ->
      match Metrics.max_drawdown navs with
      | Ok (_, dur) -> dur >= 0 && dur <= Array.length navs - 1
      | Error _ -> false)

(** {b Sample standard deviation is non-negative.} *)
let prop_stddev_non_negative =
  QCheck.Test.make ~count:n_runs ~name:"sample stddev is non-negative"
    finite_returns (fun xs ->
      match Metrics.stddev xs with Ok sd -> sd >= 0. | Error _ -> false)

(** {b Standard deviation is shift-invariant.}

    Adding a constant to every observation moves the mean by that constant and
    leaves every deviation unchanged. A failure here would indicate the
    two-pass computation had regressed to an uncentered form. *)
let prop_stddev_shift_invariant =
  QCheck.Test.make ~count:n_runs ~name:"stddev is shift-invariant"
    QCheck.(pair finite_returns (float_range (-100.) 100.))
    (fun (xs, c) ->
      let shifted = Array.map (fun x -> x +. c) xs in
      match (Metrics.stddev xs, Metrics.stddev shifted) with
      | Ok a, Ok b -> Float.abs (a -. b) < 1e-6 *. (1. +. Float.abs a)
      | _ -> false)

(** {b Standard deviation scales linearly.}

    [sd(k*x) = |k| * sd(x)]. *)
let prop_stddev_scales_linearly =
  QCheck.Test.make ~count:n_runs ~name:"stddev scales linearly"
    QCheck.(pair finite_returns (float_range 0.1 100.))
    (fun (xs, k) ->
      let scaled = Array.map (fun x -> x *. k) xs in
      match (Metrics.stddev xs, Metrics.stddev scaled) with
      | Ok a, Ok b -> Float.abs (b -. (k *. a)) < 1e-6 *. (1. +. Float.abs b)
      | _ -> false)

(** {b The mean is bounded by the extremes.} *)
let prop_mean_within_range =
  QCheck.Test.make ~count:n_runs ~name:"mean lies between min and max"
    finite_returns (fun xs ->
      match Metrics.mean xs with
      | Error _ -> false
      | Ok mu ->
          let lo = Array.fold_left Float.min infinity xs in
          let hi = Array.fold_left Float.max neg_infinity xs in
          mu >= lo -. 1e-9 && mu <= hi +. 1e-9)

(** {b Sharpe has the same sign as the excess mean return.}

    Since the denominator (a standard deviation) is non-negative, the sign of
    the Sharpe ratio is the sign of the mean excess return. A test that fails
    here would indicate an absolute value crept into the numerator — exactly
    the bug that would turn a losing strategy into a winning-looking one. *)
let prop_sharpe_sign_matches_mean_excess =
  QCheck.Test.make ~count:n_runs
    ~name:"Sharpe has the sign of the mean excess return" finite_returns
    (fun returns ->
      let bars_per_year = 252. in
      let rf = 0.04 in
      let rf_bar = Float.pow (1. +. rf) (1. /. bars_per_year) -. 1. in
      match
        ( Metrics.sharpe ~returns ~risk_free_annual:rf ~bars_per_year,
          Metrics.mean (Array.map (fun r -> r -. rf_bar) returns) )
      with
      | Ok s, Ok mu ->
          (* When volatility is degenerate the implementation reports 0, which
             is consistent with any sign. *)
          if Float.abs s < 1e-12 then true
          else (s > 0.) = (mu > 0.)
      | _ -> true)

(** {b Sharpe is invariant to leverage.}

    Scaling every return by a positive constant scales both the mean and the
    standard deviation by that constant, so the ratio is unchanged. This holds
    only at a zero risk-free rate — with a non-zero rate, leverage changes the
    excess return, which is the correct economics and not a bug. *)
let prop_sharpe_leverage_invariant_at_zero_rf =
  QCheck.Test.make ~count:n_runs
    ~name:"Sharpe is leverage-invariant when rf = 0"
    QCheck.(pair finite_returns (float_range 0.1 10.))
    (fun (returns, k) ->
      let scaled = Array.map (fun r -> r *. k) returns in
      match
        ( Metrics.sharpe ~returns ~risk_free_annual:0. ~bars_per_year:252.,
          Metrics.sharpe ~returns:scaled ~risk_free_annual:0.
            ~bars_per_year:252. )
      with
      | Ok a, Ok b -> Float.abs (a -. b) < 1e-6 *. (1. +. Float.abs a)
      | _ -> false)

(** {b Simple returns round-trip.}

    Reconstructing NAV by compounding the returns must recover the original
    path, confirming the return definition and the NAV series are consistent. *)
let prop_returns_round_trip =
  QCheck.Test.make ~count:n_runs ~name:"NAV reconstructs from its returns"
    positive_navs (fun navs ->
      match Rolling.simple_returns navs with
      | Error _ -> false
      | Ok rs ->
          let recon = ref navs.(0) in
          let ok = ref true in
          Array.iter
            (fun r ->
              recon := !recon *. (1. +. r);
              if not (Float.is_finite !recon) then ok := false)
            rs;
          !ok
          && Float.abs (!recon -. navs.(Array.length navs - 1))
             < 1e-6 *. (1. +. Float.abs navs.(Array.length navs - 1)))

(** {b A causal view never exposes an index beyond its bound.}

    The mechanism underlying the entire no-lookahead guarantee, checked over
    random array lengths and bounds. *)
let prop_causal_view_bounds =
  QCheck.Test.make ~count:n_runs ~name:"a causal view never exposes the future"
    QCheck.(pair (int_range 1 200) (int_range 0 400))
    (fun (n, raw_t) ->
      let xs = Array.init n float_of_int in
      let t = raw_t mod n in
      let v = Causal.create xs t in
      let ok = ref true in
      (* Everything at or before t is visible. *)
      for i = 0 to t do
        if Causal.get v i = None then ok := false
      done;
      (* Nothing after t is. *)
      for i = t + 1 to n - 1 do
        if Causal.get v i <> None then ok := false
      done;
      !ok && Causal.length v = t + 1)

(** {b A lookback window always ends at the current bar.} *)
let prop_lookback_ends_at_now =
  QCheck.Test.make ~count:n_runs ~name:"a lookback window ends at the current bar"
    QCheck.(triple (int_range 1 200) (int_range 0 400) (int_range 1 50))
    (fun (n, raw_t, k) ->
      let xs = Array.init n float_of_int in
      let t = raw_t mod n in
      let v = Causal.create xs t in
      match Causal.lookback v k with
      | Error _ -> true (* insufficient data is a valid outcome *)
      | Ok w ->
          Array.length w = k
          && w.(k - 1) = float_of_int t
          && Array.for_all (fun x -> x <= float_of_int t) w)

(** {b Rolling variance is non-negative.} *)
let prop_rolling_variance_non_negative =
  QCheck.Test.make ~count:n_runs ~name:"rolling variance is non-negative"
    QCheck.(
      pair (list_size Gen.(int_range 5 100) (float_range (-100.) 100.))
        (int_range 2 20))
    (fun (xs, k) ->
      let arr = Array.of_list xs in
      let v = Causal.create arr (Array.length arr - 1) in
      match Rolling.variance v k with
      | Ok var -> var >= -1e-12
      | Error _ -> true)

(** {b OLS on an exactly linear relationship recovers the slope.}

    For any non-degenerate [x] and any [(a, b)], regressing [y = a + b*x] on
    [x] must return exactly [b]. *)
let prop_ols_recovers_exact_slope =
  QCheck.Test.make ~count:n_runs
    ~name:"OLS recovers the slope of an exactly linear relationship"
    QCheck.(pair (float_range (-10.) 10.) (float_range (-10.) 10.))
    (fun (a, b) ->
      let n = 30 in
      (* A regressor with genuine variance; a constant one is correctly
         refused and is covered by an example-based test instead. *)
      let x = Array.init n (fun i -> float_of_int i +. (0.5 *. sin (float_of_int i))) in
      let y = Array.map (fun xi -> a +. (b *. xi)) x in
      let v = n - 1 in
      match
        Ols.fit_window ~y:(Causal.create y v) ~x:(Causal.create x v) n
      with
      | Error _ -> false
      | Ok fit ->
          Float.abs (fit.beta -. b) < 1e-6 *. (1. +. Float.abs b)
          && Float.abs (fit.alpha -. a) < 1e-6 *. (1. +. Float.abs a))

(** {b The backtest never produces a NaN or infinite NAV.}

    Over randomly parameterised cointegrated series, every NAV must be finite
    and positive. A NaN escaping into the NAV series would silently poison
    every downstream metric. *)
let prop_backtest_nav_always_finite =
  QCheck.Test.make ~count:60
    ~name:"backtest NAV is always finite and positive"
    QCheck.(triple (int_range 1 10000) (float_range 0.5 2.0) (float_range 5. 30.))
    (fun (seed, beta, half_life) ->
      let cfg = Fixtures.test_config () in
      let series =
        Fixtures.cointegrated ~n:250 ~seed ~beta ~half_life ~sigma_spread:0.03
          ~sigma_common:0.015
      in
      match Backtest.run cfg series with
      | Error _ -> false
      | Ok res ->
          Array.for_all (fun v -> Float.is_finite v && v > 0.) res.navs)

(** {b Trade PnL decomposition holds for every trade.} *)
let prop_trade_pnl_decomposes =
  QCheck.Test.make ~count:60 ~name:"every trade satisfies net = gross - costs"
    QCheck.(int_range 1 10000)
    (fun seed ->
      let cfg = Fixtures.test_config () in
      let series =
        Fixtures.cointegrated ~n:250 ~seed ~beta:1.0 ~half_life:12.
          ~sigma_spread:0.03 ~sigma_common:0.012
      in
      match Backtest.run cfg series with
      | Error _ -> false
      | Ok res ->
          List.for_all
            (fun (t : Types.trade) ->
              Float.abs (t.pnl_net -. (t.pnl_gross -. t.costs)) < 1e-9
              && t.costs >= 0.)
            res.trades)

(** {b Higher costs never improve final NAV.}

    Monotonicity in the cost parameter. A violation would mean costs are being
    added somewhere instead of subtracted. *)
let prop_higher_costs_never_help =
  QCheck.Test.make ~count:40 ~name:"higher costs never increase final NAV"
    QCheck.(int_range 1 10000)
    (fun seed ->
      let series =
        Fixtures.cointegrated ~n:250 ~seed ~beta:1.0 ~half_life:12.
          ~sigma_spread:0.03 ~sigma_common:0.012
      in
      let run bps =
        Backtest.run
          (Fixtures.test_config ~commission_bps:bps ~slippage_bps:bps ())
          series
      in
      match (run 0., run 10.) with
      | Ok cheap, Ok dear ->
          dear.metrics.final_nav <= cheap.metrics.final_nav +. 1e-9
      | _ -> false)

(** {b Truncation invariance as a property.}

    The lookahead guarantee, checked over randomly generated series and random
    truncation points rather than the fixed ones in [test_lookahead.ml]. *)
let prop_truncation_invariance =
  QCheck.Test.make ~count:60
    ~name:"signals are invariant to truncating the future (property form)"
    QCheck.(pair (int_range 1 10000) (int_range 60 240))
    (fun (seed, t) ->
      let cfg = Fixtures.test_config () in
      let series =
        Fixtures.cointegrated ~n:250 ~seed ~beta:1.0 ~half_life:12.
          ~sigma_spread:0.03 ~sigma_common:0.012
      in
      match
        ( Backtest.signals_only cfg series,
          Backtest.signals_only cfg (Array.sub series 0 (t + 1)) )
      with
      | Ok full, Ok trunc ->
          let ok = ref true in
          for i = 0 to t do
            match (full.(i), trunc.(i)) with
            | None, None -> ()
            | Some (b1, s1, z1), Some (b2, s2, z2) ->
                if b1 <> b2 || s1 <> s2 || z1 <> z2 then ok := false
            | _ -> ok := false
          done;
          !ok
      | _ -> false)

let qcheck_tests =
  [
    prop_max_drawdown_is_non_positive;
    prop_max_drawdown_above_total_loss;
    prop_max_drawdown_scale_invariant;
    prop_monotone_path_has_no_drawdown;
    prop_drawdown_duration_in_range;
    prop_stddev_non_negative;
    prop_stddev_shift_invariant;
    prop_stddev_scales_linearly;
    prop_mean_within_range;
    prop_sharpe_sign_matches_mean_excess;
    prop_sharpe_leverage_invariant_at_zero_rf;
    prop_returns_round_trip;
    prop_causal_view_bounds;
    prop_lookback_ends_at_now;
    prop_rolling_variance_non_negative;
    prop_ols_recovers_exact_slope;
    prop_backtest_nav_always_finite;
    prop_trade_pnl_decomposes;
    prop_higher_costs_never_help;
    prop_truncation_invariance;
  ]

(** Adapt QCheck tests into Alcotest cases so the whole suite runs under one
    runner and reports one total.

    [check_cell_exn] raises [QCheck.Test.Test_fail] with the shrunk
    counterexample already formatted, so the failure message names the specific
    input that broke the property. The PRNG is seeded explicitly: a property
    suite that passes today and fails tomorrow on the same code is worse than
    useless, and CI must be reproducible. *)
let seeded_rand () = Random.State.make [| 0x5EED; 42 |]

(** [QCheck.Test.t] is an alias for the GADT [QCheck2.Test.t], defined as
    [Test : 'a cell -> t]. It hides the generator's element type so tests over
    different types can live in one list. Unpacking it here recovers the [cell]
    that [check_cell_exn] and [get_name] need. *)
let tests =
  List.map
    (fun (QCheck2.Test.Test cell) ->
      let name = QCheck.Test.get_name cell in
      ( name,
        `Quick,
        fun () ->
          try QCheck.Test.check_cell_exn ~long:false ~rand:(seeded_rand ()) cell
          with
          | QCheck.Test.Test_fail (n, msgs) ->
              Alcotest.failf "property '%s' failed:\n%s" n
                (String.concat "\n" msgs)
          | QCheck.Test.Test_error (n, arg, exn, _) ->
              Alcotest.failf "property '%s' raised on input %s: %s" n arg
                (Printexc.to_string exn) ))
    qcheck_tests
