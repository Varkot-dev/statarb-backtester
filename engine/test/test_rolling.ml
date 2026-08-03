(** Rolling statistics against hand-computed values. *)

open Statarb

let eps = 1e-10

let view_of (xs : float array) (t : int) = Causal.create xs t

(** Window [3,4,5] (the trailing 3 of [1..5] at t=4): mean 4. *)
let test_rolling_mean_hand_computed () =
  let xs = [| 1.; 2.; 3.; 4.; 5. |] in
  let v = view_of xs 4 in
  Alcotest.(check (float eps)) "trailing 3 mean" 4. (Fixtures.get_ok (Rolling.mean v 3));
  Alcotest.(check (float eps)) "trailing 5 mean" 3. (Fixtures.get_ok (Rolling.mean v 5));
  Alcotest.(check (float eps)) "trailing 1 mean is the current value" 5.
    (Fixtures.get_ok (Rolling.mean v 1));
  (* At an earlier bar the same window covers different data. *)
  let v2 = view_of xs 2 in
  Alcotest.(check (float eps)) "trailing 3 mean at t=2" 2.
    (Fixtures.get_ok (Rolling.mean v2 3))

(** Window [3,4,5]: mean 4, deviations [-1,0,1], sum of squares 2,
    sample variance 2/2 = 1, sd = 1. *)
let test_rolling_variance_hand_computed () =
  let xs = [| 1.; 2.; 3.; 4.; 5. |] in
  let v = view_of xs 4 in
  Alcotest.(check (float eps)) "trailing 3 variance" 1.
    (Fixtures.get_ok (Rolling.variance v 3));
  Alcotest.(check (float eps)) "trailing 3 stddev" 1.
    (Fixtures.get_ok (Rolling.stddev v 3));
  (* Whole series [1..5]: mean 3, deviations [-2,-1,0,1,2], SS = 10,
     sample variance 10/4 = 2.5. *)
  Alcotest.(check (float eps)) "trailing 5 variance" 2.5
    (Fixtures.get_ok (Rolling.variance v 5))

(** Window [3,4,5]: current value 5, mean 4, sd 1, so z = (5-4)/1 = 1. *)
let test_rolling_zscore_hand_computed () =
  let xs = [| 1.; 2.; 3.; 4.; 5. |] in
  let v = view_of xs 4 in
  match Fixtures.get_ok (Rolling.zscore v 3) with
  | Some z -> Alcotest.(check (float eps)) "z-score" 1. z
  | None -> Alcotest.fail "z-score should be defined here"

(** A symmetric window centred on the current value gives z = 0 only when the
    current value equals the window mean. For [2,4,3] at t: mean 3, current 3,
    so z = 0. *)
let test_rolling_zscore_at_mean_is_zero () =
  let xs = [| 2.; 4.; 3. |] in
  let v = view_of xs 2 in
  match Fixtures.get_ok (Rolling.zscore v 3) with
  | Some z -> Alcotest.(check (float eps)) "z-score at the mean" 0. z
  | None -> Alcotest.fail "z-score should be defined here"

(** A constant window has zero standard deviation, so the z-score is undefined
    rather than infinite. This is the guard that prevents a numerically flat
    patch from generating a spurious trade. *)
let test_rolling_zscore_constant_window_is_none () =
  let xs = Array.make 10 5. in
  let v = view_of xs 9 in
  Alcotest.(check bool) "z-score of a constant window is None" true
    (Fixtures.get_ok (Rolling.zscore v 5) = None)

(** Requesting a window longer than the visible history is an error, not a
    silently shortened window. *)
let test_rolling_insufficient_data () =
  let xs = [| 1.; 2.; 3. |] in
  let v = view_of xs 1 in
  match Rolling.mean v 5 with
  | Error (Types.Insufficient_data { needed = 5; got = 2 }) -> ()
  | Error e ->
      Alcotest.failf "wrong error: %s" (Types.string_of_error e)
  | Ok _ -> Alcotest.fail "expected Insufficient_data"

(** Variance needs at least two points for the (n-1) denominator to be
    meaningful. *)
let test_rolling_variance_rejects_window_of_one () =
  let xs = [| 1.; 2.; 3. |] in
  let v = view_of xs 2 in
  match Rolling.variance v 1 with
  | Error (Types.Config_error _) -> ()
  | _ -> Alcotest.fail "variance with window 1 should be a Config_error"

(** Perfectly correlated series have correlation 1; perfectly anti-correlated,
    -1. *)
let test_correlation_hand_computed () =
  let a = [| 1.; 2.; 3.; 4.; 5. |] in
  let b = [| 2.; 4.; 6.; 8.; 10. |] in
  let c = [| 10.; 8.; 6.; 4.; 2. |] in
  let va = view_of a 4 and vb = view_of b 4 and vc = view_of c 4 in
  (match Fixtures.get_ok (Rolling.correlation va vb 5) with
  | Some r -> Alcotest.(check (float 1e-9)) "perfect positive correlation" 1. r
  | None -> Alcotest.fail "correlation should be defined");
  match Fixtures.get_ok (Rolling.correlation va vc 5) with
  | Some r -> Alcotest.(check (float 1e-9)) "perfect negative correlation" (-1.) r
  | None -> Alcotest.fail "correlation should be defined"

(** A constant series has no variance, so correlation with it is undefined. *)
let test_correlation_constant_is_none () =
  let a = [| 1.; 2.; 3.; 4.; 5. |] in
  let b = Array.make 5 3. in
  let va = view_of a 4 and vb = view_of b 4 in
  Alcotest.(check bool) "correlation with a constant series is None" true
    (Fixtures.get_ok (Rolling.correlation va vb 5) = None)

(** Correlating views at different times would silently compare misaligned
    data; it is refused. *)
let test_correlation_rejects_misaligned_views () =
  let a = [| 1.; 2.; 3.; 4.; 5. |] in
  let b = [| 2.; 4.; 6.; 8.; 10. |] in
  let va = view_of a 4 and vb = view_of b 3 in
  match Rolling.correlation va vb 3 with
  | Error (Types.Misaligned_series _) -> ()
  | _ -> Alcotest.fail "misaligned views should be rejected"

(** Numerical stability: a series with a large mean and tiny variance is
    exactly the regime where the naive one-pass variance formula fails.

    Values are 1e8 + [0,1,2,3,4], so the true sample variance is 2.5 — the
    same as [0,1,2,3,4]. The one-pass form E[x^2] - E[x]^2 loses all precision
    here. *)
let test_variance_numerically_stable_with_large_mean () =
  let xs = Array.init 5 (fun i -> 1e8 +. float_of_int i) in
  let v = view_of xs 4 in
  let var = Fixtures.get_ok (Rolling.variance v 5) in
  Alcotest.(check (float 1e-6))
    "variance is unaffected by a large offset" 2.5 var

(** Shifting every value by a constant must leave the variance unchanged and
    the z-score unchanged. *)
let test_variance_is_shift_invariant () =
  let base = [| 3.; 1.; 4.; 1.; 5.; 9.; 2.; 6. |] in
  let shifted = Array.map (fun x -> x +. 1000.) base in
  let vb = view_of base 7 and vs = view_of shifted 7 in
  Alcotest.(check (float 1e-9))
    "variance is shift-invariant"
    (Fixtures.get_ok (Rolling.variance vb 8))
    (Fixtures.get_ok (Rolling.variance vs 8))

let tests =
  [
    ("rolling mean hand-computed", `Quick, test_rolling_mean_hand_computed);
    ("rolling variance hand-computed", `Quick, test_rolling_variance_hand_computed);
    ("rolling z-score hand-computed", `Quick, test_rolling_zscore_hand_computed);
    ("z-score at the mean is zero", `Quick, test_rolling_zscore_at_mean_is_zero);
    ("z-score of a constant window is None", `Quick,
     test_rolling_zscore_constant_window_is_none);
    ("insufficient data is an error", `Quick, test_rolling_insufficient_data);
    ("variance rejects window of 1", `Quick, test_rolling_variance_rejects_window_of_one);
    ("correlation hand-computed", `Quick, test_correlation_hand_computed);
    ("correlation with a constant is None", `Quick, test_correlation_constant_is_none);
    ("correlation rejects misaligned views", `Quick,
     test_correlation_rejects_misaligned_views);
    ("variance is numerically stable with a large mean", `Quick,
     test_variance_numerically_stable_with_large_mean);
    ("variance is shift-invariant", `Quick, test_variance_is_shift_invariant);
  ]
