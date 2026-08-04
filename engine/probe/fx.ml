(* Probe fixtures: copy of engine/test/fixtures.ml minus the alcotest dep. *)
open Statarb
open Statarb.Types

module Lcg = struct
  type t = { mutable state : int }

  let create seed = { state = seed land 0x3FFFFFFF }

  let next_int (g : t) : int =
    g.state <- ((1664525 * g.state) + 1013904223) land 0x3FFFFFFF;
    g.state

  let next_float (g : t) : float = float_of_int (next_int g) /. 1073741824.

  let next_gaussian (g : t) : float =
    let u1 = Float.max 1e-12 (next_float g) in
    let u2 = next_float g in
    sqrt (-2. *. log u1) *. cos (2. *. Float.pi *. u2)
end

let date_of_index (i : int) : string =
  Printf.sprintf "%04d-%02d-%02d" (2000 + (i / 372)) ((i / 31 mod 12) + 1)
    ((i mod 31) + 1)

let bar_exn ~date ~a ~b : bar =
  match (Price.of_float a, Price.of_float b) with
  | Ok pa, Ok pb -> { date; price_a = pa; price_b = pb }
  | _ -> failwith (Printf.sprintf "fixture: invalid prices %g %g" a b)

let cointegrated ~(n : int) ~(seed : int) ~(beta : float) ~(half_life : float)
    ~(sigma_spread : float) ~(sigma_common : float) : series =
  let g = Lcg.create seed in
  let phi = exp (-.log 2. /. half_life) in
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

let independent_walks ~(n : int) ~(seed : int) ~(sigma : float) : series =
  let g = Lcg.create seed in
  let la = ref (log 100.) and lb = ref (log 50.) in
  Array.init n (fun i ->
      la := !la +. (sigma *. Lcg.next_gaussian g);
      lb := !lb +. (sigma *. Lcg.next_gaussian g);
      bar_exn ~date:(date_of_index i) ~a:(exp !la) ~b:(exp !lb))

let test_config ?(hedge_window = 20) ?(zscore_window = 20)
    ?(entry_threshold = 2.0) ?(exit_threshold = 0.5)
    ?(stop_loss_threshold = 3.5) ?(max_holding_bars = 30)
    ?(commission_bps = 1.0) ?(slippage_bps = 2.0) ?(initial_capital = 100_000.)
    ?(capital_per_trade_frac = 0.25) ?(bars_per_year = 252.)
    ?(risk_free_rate = 0.04) ?(accrue_cash_interest = true) () : Config.t =
  match
    Config.create ~hedge_window ~zscore_window ~entry_threshold ~exit_threshold
      ~stop_loss_threshold ~max_holding_bars ~commission_bps ~slippage_bps
      ~initial_capital ~capital_per_trade_frac ~bars_per_year ~risk_free_rate
      ~accrue_cash_interest ()
  with
  | Ok c -> c
  | Error e -> failwith ("test_config invalid: " ^ string_of_error e)

let zero_cost_config () : Config.t = test_config ~commission_bps:0. ~slippage_bps:0. ()

let get_ok = function
  | Ok x -> x
  | Error e -> failwith ("expected Ok, got Error: " ^ string_of_error e)

(* --- probe reporting helpers --- *)
let hdr s =
  Printf.printf "\n========================================================\n";
  Printf.printf "%s\n" s;
  Printf.printf "========================================================\n"

let n_fail = ref 0

let ck label ok =
  if ok then Printf.printf "  [ok]   %s\n" label
  else begin
    incr n_fail;
    Printf.printf "  [FAIL] %s\n" label
  end

let ckf label expected got tol =
  let d = Float.abs (expected -. got) in
  if d <= tol && Float.is_finite d then
    Printf.printf "  [ok]   %s (exp %.10g got %.10g, diff %.3e)\n" label expected got d
  else begin
    incr n_fail;
    Printf.printf "  [FAIL] %s (exp %.10g got %.10g, diff %.3e)\n" label expected got d
  end
