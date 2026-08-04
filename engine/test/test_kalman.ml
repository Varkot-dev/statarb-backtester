(** Kalman filter validation.

    The claim being checked is not "the filter runs" but "the filter recovers a
    hidden parameter it was never shown, and does so better than the rolling-OLS
    alternative on data where the alternative is known to struggle." Everything
    here is measured against generated ground truth. *)

open Statarb
open Statarb.Types

let params = Kalman.default_params

(** A pair whose true hedge ratio drifts linearly. Rolling OLS assumes beta is
    constant over its window, so this is precisely the regime where it should
    lag and the filter should not. *)
let drifting_beta_series ~(n : int) ~(seed : int) ~(beta_start : float)
    ~(beta_end : float) : series * (int -> float) =
  let g = Fixtures.Lcg.create seed in
  let true_beta i =
    beta_start
    +. ((beta_end -. beta_start) *. float_of_int i /. float_of_int n)
  in
  let phi = exp (-.log 2. /. 15.) in
  let innov = 0.03 *. sqrt (1. -. (phi *. phi)) in
  let c = ref 0. and s = ref 0. in
  let series =
    Array.init n (fun i ->
        c := !c +. (0.01 *. Fixtures.Lcg.next_gaussian g);
        s := (phi *. !s) +. (innov *. Fixtures.Lcg.next_gaussian g);
        let la = log 100. +. (true_beta i *. !c) +. !s in
        let lb = log 50. +. !c in
        Fixtures.bar_exn ~date:(Fixtures.date_of_index i) ~a:(exp la)
          ~b:(exp lb))
  in
  (series, true_beta)

(** {b The headline claim.}

    On a pair whose beta drifts from 1.0 to 1.6, the filter must track the true
    parameter substantially better than a 60-bar rolling regression.

    Measured as RMSE against ground truth over the whole path, not at a
    hand-picked bar. *)
let test_tracks_a_drifting_beta_better_than_rolling_ols () =
  let n = 1500 in
  let series, true_beta =
    drifting_beta_series ~n ~seed:12345 ~beta_start:1.0 ~beta_end:1.6
  in
  let log_a = Array.map (fun b -> log (Price.to_float b.price_a)) series in
  let log_b = Array.map (fun b -> log (Price.to_float b.price_b)) series in
  let filtered = Kalman.run params series in

  let sq_err_ols = ref 0. and sq_err_kalman = ref 0. and counted = ref 0 in
  for i = 100 to n - 1 do
    let truth = true_beta i in
    (match
       Ols.fit_window ~y:(Causal.create log_a i) ~x:(Causal.create log_b i) 60
     with
    | Ok fit ->
        sq_err_ols := !sq_err_ols +. ((fit.beta -. truth) ** 2.);
        sq_err_kalman :=
          !sq_err_kalman +. ((filtered.(i).new_state.beta -. truth) ** 2.);
        incr counted
    | Error _ -> ())
  done;
  let count = float_of_int !counted in
  let rmse_ols = sqrt (!sq_err_ols /. count) in
  let rmse_kalman = sqrt (!sq_err_kalman /. count) in

  Alcotest.(check bool)
    (Printf.sprintf "compared enough bars (got %d)" !counted)
    true (!counted > 1000);
  Alcotest.(check bool)
    (Printf.sprintf
       "Kalman tracks a drifting beta better than rolling OLS (RMSE %.5f vs \
        %.5f)"
       rmse_kalman rmse_ols)
    true
    (rmse_kalman < rmse_ols *. 0.75)

(** On a {e constant} beta the filter must converge to it.

    The complement of the test above: tracking ability is worthless if it comes
    at the cost of never settling. *)
let test_converges_on_a_constant_beta () =
  let true_beta = 1.35 in
  let series, _ =
    drifting_beta_series ~n:2000 ~seed:999 ~beta_start:true_beta
      ~beta_end:true_beta
  in
  let filtered = Kalman.run params series in
  let final = filtered.(Array.length filtered - 1).new_state.beta in
  Alcotest.(check (float 0.15))
    "converges to the true constant beta" true_beta final

(** State covariance must shrink as evidence accumulates: the filter should
    become more confident, and that confidence is what down-weights later
    observations. A [P] that grew without bound would mean the filter never
    learns. *)
let test_uncertainty_shrinks_with_evidence () =
  let series, _ =
    drifting_beta_series ~n:500 ~seed:7 ~beta_start:1.2 ~beta_end:1.2
  in
  let filtered = Kalman.run params series in
  let early = filtered.(20).new_state.p11 in
  let late = filtered.(400).new_state.p11 in
  Alcotest.(check bool)
    (Printf.sprintf "beta variance falls with evidence (%.3e -> %.3e)" early late)
    true (late < early);
  Alcotest.(check bool) "variance stays non-negative" true (late >= 0.)

(** {b Causality.}

    The recursion cannot see the future — but the same truncation test applied
    everywhere else in this codebase is run here too, because "it is obviously
    causal" is exactly the reasoning that lets bugs through. *)
let test_filter_is_causal_under_truncation () =
  let series, _ =
    drifting_beta_series ~n:600 ~seed:31 ~beta_start:1.0 ~beta_end:1.4
  in
  let full = Kalman.run params series in
  List.iter
    (fun t ->
      let truncated = Kalman.run params (Array.sub series 0 (t + 1)) in
      for i = 0 to t do
        Alcotest.(check (float 0.))
          (Printf.sprintf "beta at bar %d unchanged by truncation at %d" i t)
          full.(i).new_state.beta truncated.(i).new_state.beta;
        Alcotest.(check (float 0.))
          (Printf.sprintf "innovation at bar %d unchanged by truncation at %d" i t)
          full.(i).innovation truncated.(i).innovation
      done)
    [ 100; 250; 400; 599 ]

(** The innovation is a genuine one-step-ahead prediction error, so on
    well-specified data it should be roughly zero-mean. A strongly biased
    innovation would mean the model is systematically mis-predicting. *)
let test_innovations_are_approximately_zero_mean () =
  let series, _ =
    drifting_beta_series ~n:1500 ~seed:77 ~beta_start:1.1 ~beta_end:1.1
  in
  let filtered = Kalman.run params series in
  let tail =
    Array.sub filtered 200 (Array.length filtered - 200)
    |> Array.map (fun u -> u.Kalman.innovation)
  in
  let mean = Fixtures.get_ok (Metrics.mean tail) in
  let sd = Fixtures.get_ok (Metrics.stddev tail) in
  Alcotest.(check bool)
    (Printf.sprintf "innovations are near zero-mean (mean %.5f, sd %.5f)" mean sd)
    true
    (Float.abs mean < 0.5 *. sd)

(** The standardised innovation is a z-score with no window parameter — the
    filter supplies its own scale. It should behave like one: mostly inside
    ±3, and with a standard deviation of order 1. *)
let test_standardised_innovation_behaves_like_a_zscore () =
  let series, _ =
    drifting_beta_series ~n:1500 ~seed:41 ~beta_start:1.2 ~beta_end:1.2
  in
  let filtered = Kalman.run params series in
  let z =
    Array.sub filtered 200 (Array.length filtered - 200)
    |> Array.map (fun u -> u.Kalman.standardised_innovation)
  in
  let extreme =
    Array.fold_left (fun acc v -> if Float.abs v > 3.0 then acc + 1 else acc) 0 z
  in
  let fraction = float_of_int extreme /. float_of_int (Array.length z) in
  Alcotest.(check bool)
    (Printf.sprintf "few |z| > 3 (%.1f%%)" (fraction *. 100.))
    true (fraction < 0.10);
  Array.iter
    (fun v ->
      if not (Float.is_finite v) then
        Alcotest.fail "standardised innovation is not finite")
    z

(** Signal-to-noise controls responsiveness, and the effective-memory helper
    must report the correspondence honestly. *)
let test_signal_to_noise_controls_responsiveness () =
  let series, true_beta =
    drifting_beta_series ~n:1200 ~seed:5 ~beta_start:1.0 ~beta_end:1.8
  in
  let fast =
    Fixtures.get_ok
      (Kalman.make_params ~observation_variance:1e-3 ~state_variance:1e-6 ())
  in
  let slow =
    Fixtures.get_ok
      (Kalman.make_params ~observation_variance:1e-3 ~state_variance:1e-10 ())
  in
  Alcotest.(check bool) "faster filter has a shorter memory" true
    (Kalman.effective_memory_bars fast < Kalman.effective_memory_bars slow);

  (* The responsive filter should end nearer the (moved) truth. *)
  let final_of p =
    let r = Kalman.run p series in
    r.(Array.length r - 1).new_state.beta
  in
  let truth = true_beta 1199 in
  Alcotest.(check bool)
    (Printf.sprintf
       "a higher Q/R tracks the drift better (fast %.3f vs slow %.3f, truth \
        %.3f)"
       (final_of fast) (final_of slow) truth)
    true
    (Float.abs (final_of fast -. truth) < Float.abs (final_of slow -. truth))

let test_params_validation () =
  List.iter
    (fun (o, s, label) ->
      match
        Kalman.make_params ~observation_variance:o ~state_variance:s ()
      with
      | Error (Config_error _) -> ()
      | _ -> Alcotest.failf "%s should be rejected" label)
    [
      (0., 1e-8, "zero observation variance");
      (-1., 1e-8, "negative observation variance");
      (1e-3, -1., "negative state variance");
    ];
  Alcotest.(check bool) "valid params are accepted" true
    (Result.is_ok (Kalman.make_params ()))

(** A zero state variance means "beta never moves", which should behave like an
    expanding-window regression rather than diverging. *)
let test_zero_state_variance_is_stable () =
  let series, _ =
    drifting_beta_series ~n:800 ~seed:13 ~beta_start:1.25 ~beta_end:1.25
  in
  let p =
    Fixtures.get_ok
      (Kalman.make_params ~observation_variance:1e-3 ~state_variance:0. ())
  in
  let filtered = Kalman.run p series in
  Array.iteri
    (fun i u ->
      if not (Float.is_finite u.Kalman.new_state.Kalman.beta) then
        Alcotest.failf "beta became non-finite at bar %d" i)
    filtered;
  Alcotest.(check bool) "infinite effective memory at Q=0" true
    (Kalman.effective_memory_bars p = infinity)

let tests =
  [
    ("tracks a drifting beta better than rolling OLS", `Quick,
     test_tracks_a_drifting_beta_better_than_rolling_ols);
    ("converges on a constant beta", `Quick, test_converges_on_a_constant_beta);
    ("uncertainty shrinks with evidence", `Quick,
     test_uncertainty_shrinks_with_evidence);
    ("filter is causal under truncation", `Quick,
     test_filter_is_causal_under_truncation);
    ("innovations are approximately zero-mean", `Quick,
     test_innovations_are_approximately_zero_mean);
    ("standardised innovation behaves like a z-score", `Quick,
     test_standardised_innovation_behaves_like_a_zscore);
    ("signal-to-noise controls responsiveness", `Quick,
     test_signal_to_noise_controls_responsiveness);
    ("params validation", `Quick, test_params_validation);
    ("zero state variance is stable", `Quick, test_zero_state_variance_is_stable);
  ]
