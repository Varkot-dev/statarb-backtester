(** Metrics verified against hand-computed fixtures.

    Each test states the arithmetic explicitly in its docstring so a reader can
    check the expected value without running anything. Where a metric has a
    convention choice (sample vs population stddev, geometric vs linear
    de-annualization) the fixture is computed under the documented convention,
    so a future change to that convention breaks the test rather than silently
    changing the README's numbers. *)

open Statarb

let eps = 1e-9
let eps_loose = 1e-6

(** {1 Mean and standard deviation} *)

(** [1,2,3,4,5]: mean 3. Sample sd = sqrt(sum of squared deviations / 4)
    = sqrt((4+1+0+1+4)/4) = sqrt(2.5) = 1.5811388300841898. *)
let test_mean_stddev_fixture () =
  let xs = [| 1.; 2.; 3.; 4.; 5. |] in
  Alcotest.(check (float eps)) "mean" 3. (Fixtures.get_ok (Metrics.mean xs));
  Alcotest.(check (float eps))
    "sample stddev" (sqrt 2.5)
    (Fixtures.get_ok (Metrics.stddev xs))

(** A constant series has zero variance. *)
let test_stddev_constant_series () =
  let xs = Array.make 10 7.5 in
  Alcotest.(check (float eps)) "stddev of constant" 0.
    (Fixtures.get_ok (Metrics.stddev xs))

(** Sample (n-1) not population (n): for [1,3] the sample sd is
    sqrt(((1-2)^2 + (3-2)^2)/1) = sqrt(2) = 1.41421356..., whereas the
    population sd would be 1. This distinguishes the two conventions. *)
let test_stddev_is_sample_not_population () =
  let xs = [| 1.; 3. |] in
  Alcotest.(check (float eps)) "sample stddev of [1;3]" (sqrt 2.)
    (Fixtures.get_ok (Metrics.stddev xs))

let test_stddev_requires_two_points () =
  match Metrics.stddev [| 1. |] with
  | Error (Types.Insufficient_data { needed = 2; got = 1 }) -> ()
  | _ -> Alcotest.fail "stddev of a single point should be Insufficient_data"

(** {1 Sharpe ratio} *)

(** Hand-computed with zero risk-free rate.

    Returns [0.01, 0.02, -0.01, 0.03, 0.00].
    mean = 0.05/5 = 0.01.
    deviations: 0, 0.01, -0.02, 0.02, -0.01
    squared:    0, 1e-4, 4e-4, 4e-4, 1e-4  -> sum 1.0e-3
    sample var = 1.0e-3 / 4 = 2.5e-4 -> sd = 0.015811388300841896
    per-bar Sharpe = 0.01 / 0.0158113883... = 0.6324555320336759
    annualized (x sqrt 252 = 15.874507866387544) = 10.03897... *)
let test_sharpe_hand_computed_zero_rf () =
  let returns = [| 0.01; 0.02; -0.01; 0.03; 0.00 |] in
  let sd = sqrt 2.5e-4 in
  let expected_per_bar = 0.01 /. sd in
  let expected_annual = expected_per_bar *. sqrt 252. in
  Alcotest.(check (float eps))
    "annualized Sharpe, rf=0" expected_annual
    (Fixtures.get_ok
       (Metrics.sharpe ~returns ~risk_free_annual:0. ~bars_per_year:252.));
  Alcotest.(check (float eps))
    "per-bar Sharpe, rf=0" expected_per_bar
    (Fixtures.get_ok
       (Metrics.sharpe_per_bar ~returns ~risk_free_annual:0.
          ~bars_per_year:252.))

(** With a non-zero risk-free rate the excess mean shifts by exactly the
    de-annualized rate, and — because the rate is a constant — the standard
    deviation is unchanged.

    rf_bar = 1.04^(1/252) - 1. Uses the geometric convention, which is what
    {!Metrics.sharpe} documents. *)
let test_sharpe_with_risk_free_rate () =
  let returns = [| 0.01; 0.02; -0.01; 0.03; 0.00 |] in
  let rf_bar = Float.pow 1.04 (1. /. 252.) -. 1. in
  let sd = sqrt 2.5e-4 in
  let expected = (0.01 -. rf_bar) /. sd *. sqrt 252. in
  Alcotest.(check (float eps))
    "annualized Sharpe, rf=4%" expected
    (Fixtures.get_ok
       (Metrics.sharpe ~returns ~risk_free_annual:0.04 ~bars_per_year:252.));
  (* And the de-annualization is geometric, not linear: assert the two differ,
     so a silent switch to rf/252 would fail this test. *)
  let linear_rf = 0.04 /. 252. in
  Alcotest.(check bool) "geometric rf differs from linear rf" true
    (Float.abs (rf_bar -. linear_rf) > 1e-9)

(** A constant return series has zero volatility. Rather than dividing by zero
    and reporting infinity, Sharpe is defined as 0. *)
let test_sharpe_zero_volatility () =
  let returns = Array.make 20 0.001 in
  Alcotest.(check (float eps))
    "Sharpe of a zero-volatility series is 0" 0.
    (Fixtures.get_ok
       (Metrics.sharpe ~returns ~risk_free_annual:0. ~bars_per_year:252.))

(** A strategy that loses money must report a negative Sharpe. This is the
    honesty check: nothing in the implementation takes an absolute value. *)
let test_sharpe_negative_for_losing_strategy () =
  let returns = [| -0.01; -0.02; 0.005; -0.015; -0.005 |] in
  let s =
    Fixtures.get_ok
      (Metrics.sharpe ~returns ~risk_free_annual:0. ~bars_per_year:252.)
  in
  Alcotest.(check bool)
    (Printf.sprintf "Sharpe of a losing strategy is negative (got %.6f)" s)
    true (s < 0.)

(** {1 Maximum drawdown} *)

(** NAV [100, 120, 90, 110, 80, 130].

    Running peak: 100, 120, 120, 120, 120, 130.
    Drawdowns:      0,   0, -0.25, -1/12, -1/3, 0.
    Worst is at NAV 80 against peak 120: 80/120 - 1 = -1/3.

    Duration: the peak preceding the worst trough is index 1 (NAV 120). NAV
    first regains 120 at index 5 (NAV 130). Duration = 5 - 1 = 4. *)
let test_max_drawdown_hand_computed () =
  let navs = [| 100.; 120.; 90.; 110.; 80.; 130. |] in
  let dd, dur = Fixtures.get_ok (Metrics.max_drawdown navs) in
  Alcotest.(check (float eps)) "max drawdown" (-1. /. 3.) dd;
  Alcotest.(check int) "drawdown duration" 4 dur

(** A monotonically rising series never draws down. *)
let test_max_drawdown_monotone_up () =
  let navs = [| 100.; 101.; 102.; 103.; 110. |] in
  let dd, dur = Fixtures.get_ok (Metrics.max_drawdown navs) in
  Alcotest.(check (float eps)) "no drawdown" 0. dd;
  Alcotest.(check int) "zero duration" 0 dur

(** A monotonically falling series is in drawdown from the first bar and never
    recovers, so the duration runs to the end of the sample. *)
let test_max_drawdown_monotone_down () =
  let navs = [| 100.; 90.; 80.; 70.; 50. |] in
  let dd, dur = Fixtures.get_ok (Metrics.max_drawdown navs) in
  Alcotest.(check (float eps)) "drawdown to 50%" (-0.5) dd;
  Alcotest.(check int) "duration runs to end of sample" 4 dur

(** Sign convention: max drawdown is reported as a negative fraction. *)
let test_max_drawdown_sign_convention () =
  let navs = [| 100.; 50. |] in
  let dd, _ = Fixtures.get_ok (Metrics.max_drawdown navs) in
  Alcotest.(check bool) "max drawdown is negative" true (dd < 0.);
  Alcotest.(check (float eps)) "50% loss is -0.5" (-0.5) dd

(** A NAV that reaches zero or below means the account was wiped out; the
    drawdown ratio is not meaningful and the function refuses rather than
    returning a misleading number. *)
let test_max_drawdown_rejects_nonpositive_nav () =
  match Metrics.max_drawdown [| 100.; 0. |] with
  | Error (Types.Invalid_price _) -> ()
  | _ -> Alcotest.fail "max_drawdown should reject a non-positive NAV"

(** A later, deeper drawdown from a higher peak must win over an earlier
    shallower one — the metric is a maximum over the whole path, not the
    first drawdown found. *)
let test_max_drawdown_picks_the_worst_not_the_first () =
  (* First drawdown: 100 -> 95 = -5%. Second: 200 -> 150 = -25%. *)
  let navs = [| 100.; 95.; 200.; 150.; 210. |] in
  let dd, _ = Fixtures.get_ok (Metrics.max_drawdown navs) in
  Alcotest.(check (float eps)) "worst drawdown is the later, deeper one" (-0.25) dd

(** {1 Annualized return} *)

(** Doubling over exactly one year (252 bars at 252 bars/year) is +100%. *)
let test_annualized_return_one_year_double () =
  let r =
    Fixtures.get_ok
      (Metrics.annualized_return ~initial:100. ~final:200. ~n_bars:252
         ~bars_per_year:252.)
  in
  Alcotest.(check (float eps_loose)) "doubling in one year" 1.0 r

(** Doubling over two years is 2^(1/2) - 1 = 41.42%, not 50%: the metric is
    geometric, not linear. *)
let test_annualized_return_is_geometric () =
  let r =
    Fixtures.get_ok
      (Metrics.annualized_return ~initial:100. ~final:200. ~n_bars:504
         ~bars_per_year:252.)
  in
  Alcotest.(check (float eps_loose))
    "doubling over two years" (sqrt 2. -. 1.) r;
  Alcotest.(check bool) "differs from the linear answer of 0.5" true
    (Float.abs (r -. 0.5) > 0.05)

(** A loss annualizes to a negative number. *)
let test_annualized_return_negative () =
  let r =
    Fixtures.get_ok
      (Metrics.annualized_return ~initial:100. ~final:50. ~n_bars:252
         ~bars_per_year:252.)
  in
  Alcotest.(check (float eps_loose)) "halving in one year" (-0.5) r

(** {1 Calmar} *)

(** Calmar = annualized return / |max drawdown| = 0.20 / 0.10 = 2.0. *)
let test_calmar_hand_computed () =
  Alcotest.(check (float eps))
    "calmar" 2.0
    (Metrics.calmar ~annualized_return:0.20 ~max_drawdown:(-0.10))

(** No drawdown means Calmar is reported as 0 rather than infinity. *)
let test_calmar_no_drawdown () =
  Alcotest.(check (float eps))
    "calmar with zero drawdown" 0.
    (Metrics.calmar ~annualized_return:0.20 ~max_drawdown:0.)

(** {1 Returns} *)

(** NAV [100, 110, 99] -> returns [0.10, -0.10]. *)
let test_simple_returns_hand_computed () =
  let navs = [| 100.; 110.; 99. |] in
  let rs = Fixtures.get_ok (Rolling.simple_returns navs) in
  Alcotest.(check int) "two returns from three NAVs" 2 (Array.length rs);
  Alcotest.(check (float eps)) "first return" 0.10 rs.(0);
  Alcotest.(check (float eps)) "second return" (-0.10) rs.(1)

let tests =
  [
    ("mean and sample stddev fixture", `Quick, test_mean_stddev_fixture);
    ("stddev of a constant series", `Quick, test_stddev_constant_series);
    ("stddev uses n-1 not n", `Quick, test_stddev_is_sample_not_population);
    ("stddev requires two points", `Quick, test_stddev_requires_two_points);
    ("Sharpe hand-computed, rf=0", `Quick, test_sharpe_hand_computed_zero_rf);
    ("Sharpe with risk-free rate (geometric)", `Quick, test_sharpe_with_risk_free_rate);
    ("Sharpe with zero volatility is 0", `Quick, test_sharpe_zero_volatility);
    ("Sharpe is negative for a losing strategy", `Quick,
     test_sharpe_negative_for_losing_strategy);
    ("max drawdown hand-computed", `Quick, test_max_drawdown_hand_computed);
    ("max drawdown, monotone up", `Quick, test_max_drawdown_monotone_up);
    ("max drawdown, monotone down", `Quick, test_max_drawdown_monotone_down);
    ("max drawdown sign convention", `Quick, test_max_drawdown_sign_convention);
    ("max drawdown rejects non-positive NAV", `Quick,
     test_max_drawdown_rejects_nonpositive_nav);
    ("max drawdown picks the worst, not the first", `Quick,
     test_max_drawdown_picks_the_worst_not_the_first);
    ("annualized return, 1y double", `Quick, test_annualized_return_one_year_double);
    ("annualized return is geometric", `Quick, test_annualized_return_is_geometric);
    ("annualized return, negative", `Quick, test_annualized_return_negative);
    ("calmar hand-computed", `Quick, test_calmar_hand_computed);
    ("calmar with no drawdown", `Quick, test_calmar_no_drawdown);
    ("simple returns hand-computed", `Quick, test_simple_returns_hand_computed);
  ]
