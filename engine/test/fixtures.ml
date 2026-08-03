(** Deterministic test data.

    Everything here is seeded and reproducible. No test in this suite depends
    on wall-clock time, on the system PRNG, or on data fetched from a network. *)

open Statarb
open Statarb.Types

(** A small, portable linear congruential generator.

    We do not use [Random] from the stdlib because its exact output is not
    guaranteed stable across OCaml versions; a test asserting a specific number
    of trades would then break on a compiler upgrade for reasons unrelated to
    the code under test. The constants are Numerical Recipes'. *)
module Lcg = struct
  type t = { mutable state : int }

  let create seed = { state = seed land 0x3FFFFFFF }

  let next_int (g : t) : int =
    g.state <- ((1664525 * g.state) + 1013904223) land 0x3FFFFFFF;
    g.state

  (** Uniform in [0, 1). *)
  let next_float (g : t) : float =
    float_of_int (next_int g) /. 1073741824.

  (** Standard normal via Box-Muller. Returns one variate per call, discarding
      the second; simplicity matters more than efficiency here. *)
  let next_gaussian (g : t) : float =
    let u1 = Float.max 1e-12 (next_float g) in
    let u2 = next_float g in
    sqrt (-2. *. log u1) *. cos (2. *. Float.pi *. u2)
end

let date_of_index (i : int) : string =
  (* Synthetic sequential dates. Not real calendar dates — the engine treats
     dates as opaque labels and never does date arithmetic — but they are
     strictly increasing, which the CSV loader requires. *)
  Printf.sprintf "%04d-%02d-%02d" (2000 + (i / 372)) ((i / 31 mod 12) + 1)
    ((i mod 31) + 1)

let bar_exn ~date ~a ~b : bar =
  match (Price.of_float a, Price.of_float b) with
  | Ok pa, Ok pb -> { date; price_a = pa; price_b = pb }
  | _ -> failwith (Printf.sprintf "fixture: invalid prices %g %g" a b)

(** [cointegrated ~n ~seed ~beta ~half_life ~sigma_spread ~sigma_common]
    generates a genuinely cointegrated pair with {e known} ground-truth
    parameters, so correctness can be checked against the truth rather than
    against the engine's own output.

    Construction:
    - A common stochastic trend [c(t)] is a random walk. Both legs load on it,
      so each leg individually is I(1) — non-stationary, as real prices are.
    - The spread [s(t)] follows a mean-reverting Ornstein-Uhlenbeck process
      discretised as an AR(1) with [phi = exp(-ln 2 / half_life)], so its
      half-life of mean reversion is exactly [half_life] bars.
    - [log P_A = beta * c + s], [log P_B = c].

    This yields [log P_A - beta * log P_B = s], stationary by construction, so
    the true hedge ratio is [beta] and the true spread is [s]. Tests assert the
    rolling OLS recovers [beta] to within sampling error. *)
let cointegrated ~(n : int) ~(seed : int) ~(beta : float) ~(half_life : float)
    ~(sigma_spread : float) ~(sigma_common : float) : series =
  let g = Lcg.create seed in
  let phi = exp (-.log 2. /. half_life) in
  (* Stationary standard deviation of the AR(1) innovation chosen so the
     process has unconditional sd = sigma_spread. *)
  let innov_sd = sigma_spread *. sqrt (1. -. (phi *. phi)) in
  let c = ref 0. in
  let s = ref 0. in
  let base_a = log 100. and base_b = log 50. in
  Array.init n (fun i ->
      c := !c +. (sigma_common *. Lcg.next_gaussian g);
      s := (phi *. !s) +. (innov_sd *. Lcg.next_gaussian g);
      let la = base_a +. (beta *. !c) +. !s in
      let lb = base_b +. !c in
      bar_exn ~date:(date_of_index i) ~a:(exp la) ~b:(exp lb))

(** Two independent random walks: {e not} cointegrated. Used to check that the
    strategy does not manufacture profit where no relationship exists. *)
let independent_walks ~(n : int) ~(seed : int) ~(sigma : float) : series =
  let g = Lcg.create seed in
  let la = ref (log 100.) and lb = ref (log 50.) in
  Array.init n (fun i ->
      la := !la +. (sigma *. Lcg.next_gaussian g);
      lb := !lb +. (sigma *. Lcg.next_gaussian g);
      bar_exn ~date:(date_of_index i) ~a:(exp !la) ~b:(exp !lb))

(** A deterministic pair with a sinusoidal spread. Because the spread is a
    known analytic function, the z-score crossings — and hence the exact bars
    on which the engine should trade — can be reasoned about by hand. *)
let sinusoidal ~(n : int) ~(period : float) ~(amplitude : float) : series =
  Array.init n (fun i ->
      let x = float_of_int i in
      let s = amplitude *. sin (2. *. Float.pi *. x /. period) in
      let lb = log 50. in
      let la = log 100. +. s in
      bar_exn ~date:(date_of_index i) ~a:(exp la) ~b:(exp lb))

(** A test config with short windows, so tests run over small series. *)
let test_config ?(hedge_window = 20) ?(zscore_window = 20)
    ?(entry_threshold = 2.0) ?(exit_threshold = 0.5)
    ?(stop_loss_threshold = 3.5) ?(max_holding_bars = 30)
    ?(commission_bps = 1.0) ?(slippage_bps = 2.0) ?(initial_capital = 100_000.)
    ?(capital_per_trade_frac = 0.25) ?(bars_per_year = 252.)
    ?(risk_free_rate = 0.04) () : Config.t =
  match
    Config.create ~hedge_window ~zscore_window ~entry_threshold ~exit_threshold
      ~stop_loss_threshold ~max_holding_bars ~commission_bps ~slippage_bps
      ~initial_capital ~capital_per_trade_frac ~bars_per_year ~risk_free_rate
  with
  | Ok c -> c
  | Error e -> failwith ("test_config invalid: " ^ string_of_error e)

(** A zero-cost variant, for tests that need to isolate PnL from costs. *)
let zero_cost_config () : Config.t =
  test_config ~commission_bps:0. ~slippage_bps:0. ()

let get_ok = function
  | Ok x -> x
  | Error e -> Alcotest.failf "expected Ok, got Error: %s" (string_of_error e)

let get_error = function
  | Ok _ -> Alcotest.fail "expected Error, got Ok"
  | Error e -> e

(** Alcotest testable for floats with an explicit tolerance. Every float
    comparison in this suite states its tolerance rather than relying on a
    default, because the appropriate tolerance differs by two orders of
    magnitude between (say) a z-score and a dollar NAV. *)
let float_eq ~(eps : float) = Alcotest.float eps
