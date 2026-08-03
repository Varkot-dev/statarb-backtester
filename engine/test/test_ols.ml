(** Rolling OLS against exact and ground-truth fixtures. *)

open Statarb

let eps = 1e-10

(** A perfect line [y = 2x + 1] must recover alpha=1, beta=2, R^2=1 exactly. *)
let test_ols_exact_line () =
  let x = [| 1.; 2.; 3.; 4.; 5. |] in
  let y = Array.map (fun v -> (2. *. v) +. 1.) x in
  let vx = Causal.create x 4 and vy = Causal.create y 4 in
  let fit = Fixtures.get_ok (Ols.fit_window ~y:vy ~x:vx 5) in
  Alcotest.(check (float eps)) "beta" 2. fit.beta;
  Alcotest.(check (float eps)) "alpha" 1. fit.alpha;
  Alcotest.(check (float 1e-9)) "R^2 of an exact fit" 1. fit.r_squared

(** A negative slope is recovered with its sign. Beta is never abs-valued in
    the fit; direction handling happens in sizing. *)
let test_ols_negative_slope () =
  let x = [| 1.; 2.; 3.; 4.; 5. |] in
  let y = Array.map (fun v -> (-1.5 *. v) +. 10.) x in
  let vx = Causal.create x 4 and vy = Causal.create y 4 in
  let fit = Fixtures.get_ok (Ols.fit_window ~y:vy ~x:vx 5) in
  Alcotest.(check (float eps)) "negative beta" (-1.5) fit.beta;
  Alcotest.(check (float eps)) "alpha" 10. fit.alpha

(** Hand-computed with noise.

    x = [1,2,3,4], y = [2,4,5,9].
    mean x = 2.5, mean y = 5.
    dx = [-1.5,-0.5,0.5,1.5], dy = [-3,-1,0,4]
    Sxy = 4.5 + 0.5 + 0 + 6 = 11
    Sxx = 2.25 + 0.25 + 0.25 + 2.25 = 5
    beta = 11/5 = 2.2
    alpha = 5 - 2.2*2.5 = -0.5 *)
let test_ols_hand_computed_with_noise () =
  let x = [| 1.; 2.; 3.; 4. |] in
  let y = [| 2.; 4.; 5.; 9. |] in
  let vx = Causal.create x 3 and vy = Causal.create y 3 in
  let fit = Fixtures.get_ok (Ols.fit_window ~y:vy ~x:vx 4) in
  Alcotest.(check (float eps)) "beta" 2.2 fit.beta;
  Alcotest.(check (float eps)) "alpha" (-0.5) fit.alpha

(** The window is trailing: a fit at bar t uses only [t-k+1..t], so changing
    earlier data outside the window must not move the fit. *)
let test_ols_window_is_trailing () =
  let x = [| 100.; 200.; 1.; 2.; 3.; 4.; 5. |] in
  let y = Array.map (fun v -> (3. *. v) +. 2.) x in
  let vx = Causal.create x 6 and vy = Causal.create y 6 in
  (* Trailing 5 covers x = [1,2,3,4,5]; the 100/200 outliers are excluded. *)
  let fit = Fixtures.get_ok (Ols.fit_window ~y:vy ~x:vx 5) in
  Alcotest.(check (float 1e-9)) "beta from the trailing window only" 3. fit.beta

(** A constant regressor has no variance, so beta is not identified. The fit is
    refused rather than producing an enormous hedge ratio. *)
let test_ols_rejects_degenerate_regressor () =
  let x = Array.make 10 5. in
  let y = Array.init 10 float_of_int in
  let vx = Causal.create x 9 and vy = Causal.create y 9 in
  match Ols.fit_window ~y:vy ~x:vx 10 with
  | Error (Types.Degenerate_regression _) -> ()
  | _ -> Alcotest.fail "a constant regressor should be refused"

let test_ols_rejects_short_window () =
  let x = Array.init 10 float_of_int in
  let y = Array.init 10 float_of_int in
  let vx = Causal.create x 9 and vy = Causal.create y 9 in
  match Ols.fit_window ~y:vy ~x:vx 2 with
  | Error (Types.Config_error _) -> ()
  | _ -> Alcotest.fail "a window below 3 should be a Config_error"

let test_ols_rejects_misaligned_views () =
  let x = Array.init 10 float_of_int in
  let y = Array.init 10 float_of_int in
  let vx = Causal.create x 9 and vy = Causal.create y 8 in
  match Ols.fit_window ~y:vy ~x:vx 5 with
  | Error (Types.Misaligned_series _) -> ()
  | _ -> Alcotest.fail "misaligned views should be refused"

(** The residual of an exact fit is zero at every in-window point. *)
let test_residual_of_exact_fit_is_zero () =
  let x = [| 1.; 2.; 3.; 4.; 5. |] in
  let y = Array.map (fun v -> (2. *. v) +. 1.) x in
  let vx = Causal.create x 4 and vy = Causal.create y 4 in
  let fit = Fixtures.get_ok (Ols.fit_window ~y:vy ~x:vx 5) in
  Array.iteri
    (fun i xi ->
      Alcotest.(check (float 1e-9))
        (Printf.sprintf "residual at %d" i)
        0.
        (Ols.residual fit ~y_t:y.(i) ~x_t:xi))
    x

(** {b Ground truth.}

    The fixture generator constructs [log P_A = beta * c + s],
    [log P_B = c], so the true hedge ratio is exactly [beta]. Rolling OLS on a
    long window must recover it to within sampling error.

    This is the check that the estimator is measuring the right thing, as
    opposed to merely being self-consistent. *)
let test_ols_recovers_known_beta () =
  List.iter
    (fun true_beta ->
      let series =
        Fixtures.cointegrated ~n:1200 ~seed:17 ~beta:true_beta ~half_life:15.
          ~sigma_spread:0.01 ~sigma_common:0.02
      in
      let la =
        Array.map (fun b -> log (Types.Price.to_float b.Types.price_a)) series
      in
      let lb =
        Array.map (fun b -> log (Types.Price.to_float b.Types.price_b)) series
      in
      let t = 1199 in
      let fit =
        Fixtures.get_ok
          (Ols.fit_window ~y:(Causal.create la t) ~x:(Causal.create lb t) 500)
      in
      (* The spread is small relative to the common trend (sigma_spread 0.01 vs
         sigma_common 0.02 compounding over 500 bars), so the regression is
         well-identified and 5% tolerance is generous. *)
      let err = Float.abs (fit.beta -. true_beta) /. Float.abs true_beta in
      Alcotest.(check bool)
        (Printf.sprintf "recovered beta %.4f is within 5%% of the true %.4f"
           fit.beta true_beta)
        true (err < 0.05))
    [ 0.8; 1.0; 1.5 ]

(** Numerical stability with large levels: log-prices sit around 4-5 with tiny
    dispersion, which is where the uncentered normal equations cancel. Adding a
    large constant to both series must not move beta. *)
let test_ols_numerically_stable_with_offset () =
  let x = [| 1.; 2.; 3.; 4.; 5. |] in
  let y = [| 2.; 4.; 5.; 9.; 10. |] in
  let big = 1e7 in
  let xo = Array.map (fun v -> v +. big) x in
  let yo = Array.map (fun v -> v +. big) y in
  let f1 =
    Fixtures.get_ok
      (Ols.fit_window ~y:(Causal.create y 4) ~x:(Causal.create x 4) 5)
  in
  let f2 =
    Fixtures.get_ok
      (Ols.fit_window ~y:(Causal.create yo 4) ~x:(Causal.create xo 4) 5)
  in
  Alcotest.(check (float 1e-6)) "beta is offset-invariant" f1.beta f2.beta

let tests =
  [
    ("OLS on an exact line", `Quick, test_ols_exact_line);
    ("OLS recovers a negative slope", `Quick, test_ols_negative_slope);
    ("OLS hand-computed with noise", `Quick, test_ols_hand_computed_with_noise);
    ("OLS window is trailing", `Quick, test_ols_window_is_trailing);
    ("OLS rejects a degenerate regressor", `Quick, test_ols_rejects_degenerate_regressor);
    ("OLS rejects a short window", `Quick, test_ols_rejects_short_window);
    ("OLS rejects misaligned views", `Quick, test_ols_rejects_misaligned_views);
    ("residual of an exact fit is zero", `Quick, test_residual_of_exact_fit_is_zero);
    ("OLS recovers a known ground-truth beta", `Quick, test_ols_recovers_known_beta);
    ("OLS is numerically stable with a large offset", `Quick,
     test_ols_numerically_stable_with_offset);
  ]
