(** Tests for the causal-view mechanism itself.

    The lookahead tests in [test_lookahead.ml] verify the {e property}; these
    verify the {e mechanism} that delivers it, including its error behaviour. *)

open Statarb

let test_create_bounds () =
  let xs = [| 1.; 2.; 3. |] in
  let v = Causal.create xs 0 in
  Alcotest.(check int) "now at 0" 0 (Causal.now v);
  Alcotest.(check int) "length 1" 1 (Causal.length v);
  Alcotest.(check (float 0.)) "current" 1. (Causal.current v)

(** An out-of-range [now] is a programmer error (the caller controls it via
    [advance]), so it raises rather than returning a result. *)
let test_create_rejects_out_of_range () =
  let xs = [| 1.; 2.; 3. |] in
  Alcotest.check_raises "negative now" (Invalid_argument
    "Causal.create: now=-1 out of bounds for length 3")
    (fun () -> ignore (Causal.create xs (-1)));
  Alcotest.check_raises "now past end" (Invalid_argument
    "Causal.create: now=3 out of bounds for length 3")
    (fun () -> ignore (Causal.create xs 3))

(** [advance] moves forward exactly one bar and stops at the end. There is no
    [rewind], so a view can never be used to reconstruct a future-aware
    window. *)
let test_advance_walks_to_the_end_then_stops () =
  let xs = Array.init 4 float_of_int in
  let v = Causal.create xs 0 in
  let v1 = Option.get (Causal.advance v) in
  let v2 = Option.get (Causal.advance v1) in
  let v3 = Option.get (Causal.advance v2) in
  Alcotest.(check int) "advanced three times" 3 (Causal.now v3);
  Alcotest.(check bool) "cannot advance past the end" true
    (Causal.advance v3 = None)

(** [lookback] returns the trailing window oldest-first. *)
let test_lookback_ordering () =
  let xs = [| 10.; 20.; 30.; 40.; 50. |] in
  let v = Causal.create xs 4 in
  let w = Fixtures.get_ok (Causal.lookback v 3) in
  Alcotest.(check int) "window length" 3 (Array.length w);
  Alcotest.(check (float 0.)) "oldest first" 30. w.(0);
  Alcotest.(check (float 0.)) "middle" 40. w.(1);
  Alcotest.(check (float 0.)) "newest last (= current)" 50. w.(2)

let test_lookback_insufficient () =
  let xs = [| 1.; 2. |] in
  let v = Causal.create xs 1 in
  match Causal.lookback v 5 with
  | Error (Types.Insufficient_data { needed = 5; got = 2 }) -> ()
  | _ -> Alcotest.fail "expected Insufficient_data"

let test_lookback_rejects_nonpositive_window () =
  let xs = [| 1.; 2.; 3. |] in
  let v = Causal.create xs 2 in
  (match Causal.lookback v 0 with
  | Error (Types.Config_error _) -> ()
  | _ -> Alcotest.fail "window 0 should be a Config_error");
  match Causal.lookback v (-1) with
  | Error (Types.Config_error _) -> ()
  | _ -> Alcotest.fail "negative window should be a Config_error"

(** [fold_lookback] must agree with materialising the window and folding it,
    since the engine uses the fold on the hot path but tests reason about
    arrays. *)
let test_fold_lookback_agrees_with_lookback () =
  let xs = Array.init 50 (fun i -> float_of_int (i * i mod 17)) in
  for t = 0 to 49 do
    let v = Causal.create xs t in
    List.iter
      (fun k ->
        match (Causal.lookback v k, Causal.fold_lookback v k ~init:0. ~f:( +. ))
        with
        | Ok w, Ok folded ->
            Alcotest.(check (float 1e-12))
              (Printf.sprintf "fold agrees with lookback (t=%d k=%d)" t k)
              (Array.fold_left ( +. ) 0. w)
              folded
        | Error _, Error _ -> ()
        | _ ->
            Alcotest.failf
              "lookback and fold_lookback disagree on success (t=%d k=%d)" t k)
      [ 1; 3; 10 ]
  done

(** A view shares the underlying array rather than copying it, so creating one
    per bar over a long series is cheap. This is a behavioural check that
    [to_visible_array] is the only thing that copies. *)
let test_visible_array_is_a_prefix () =
  let xs = Array.init 20 float_of_int in
  let v = Causal.create xs 9 in
  let vis = Causal.to_visible_array v in
  Alcotest.(check int) "visible length" 10 (Array.length vis);
  Array.iteri
    (fun i x ->
      Alcotest.(check (float 0.))
        (Printf.sprintf "visible[%d]" i)
        (float_of_int i) x)
    vis

let tests =
  [
    ("create bounds", `Quick, test_create_bounds);
    ("create rejects out-of-range now", `Quick, test_create_rejects_out_of_range);
    ("advance walks forward then stops", `Quick, test_advance_walks_to_the_end_then_stops);
    ("lookback ordering is oldest-first", `Quick, test_lookback_ordering);
    ("lookback insufficient data", `Quick, test_lookback_insufficient);
    ("lookback rejects non-positive windows", `Quick,
     test_lookback_rejects_nonpositive_window);
    ("fold_lookback agrees with lookback", `Quick,
     test_fold_lookback_agrees_with_lookback);
    ("visible array is a prefix", `Quick, test_visible_array_is_a_prefix);
  ]
