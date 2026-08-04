open Statarb
(* THE QUESTION: my finding says rolling-OLS needs window >= 5x half-life.
   A Kalman filter has NO window. Does it escape the trap, or does Q/R
   reproduce the same pathology in different units? *)
let () =
  let n = 2520 in
  let mk hl seed =
    let st = ref seed in
    let nx () = st := (1103515245 * !st + 12345) land 0x3FFFFFFF;
                float_of_int !st /. 1073741824. in
    let g () = let u = Float.max 1e-12 (nx ()) in
               sqrt (-2. *. log u) *. cos (2. *. Float.pi *. nx ()) in
    let phi = exp (-. log 2. /. hl) in
    let iv = 0.05 *. sqrt (1. -. phi *. phi) in
    let c = ref 0. and s = ref 0. in
    Array.init n (fun i ->
      c := !c +. 0.005 *. g ();
      s := phi *. !s +. iv *. g ();
      let la = log 100. +. 1.2 *. !c +. !s and lb = log 50. +. !c in
      match Types.Price.of_float (exp la), Types.Price.of_float (exp lb) with
      | Ok a, Ok b -> { Types.date = Printf.sprintf "d%05d" i; price_a=a; price_b=b }
      | _ -> failwith "bad") in
  Printf.printf "Kalman effective-memory ÷ half-life  ->  Sharpe\n";
  Printf.printf "(if the window/half-life law generalizes, short memory = bad)\n\n";
  Printf.printf "%10s" "half_life";
  List.iter (fun r -> Printf.printf "%9s" (Printf.sprintf "mem/hl=%g" r)) [2.;5.;12.];
  Printf.printf "\n";
  List.iter (fun hl ->
    Printf.printf "%10.0f" hl;
    List.iter (fun ratio ->
      (* effective memory = 1/sqrt(Q/R) -> Q/R = 1/(ratio*hl)^2 *)
      let mem = ratio *. hl in
      let qr = 1. /. (mem *. mem) in
      let p = Result.get_ok (Kalman.make_params ~observation_variance:1e-3
                               ~state_variance:(1e-3 *. qr) ()) in
      let series = mk hl 4242 in
      let filt = Kalman.run p series in
      (* Trade the standardised innovation exactly like the z-score. *)
      let cfg = Config.default in
      let nav = ref 100000. and pos = ref 0. and qa = ref 0. and qb = ref 0. in
      let pnl = ref [] and rets = ref [] in
      Array.iteri (fun i u ->
        if i > Kalman.warmup_bars p then begin
          let pa = Types.Price.to_float series.(i).Types.price_a
          and pb = Types.Price.to_float series.(i).Types.price_b in
          let prev = !nav in
          nav := 100000. +. !qa *. pa +. !qb *. pb +. !pnl |> ignore;
          let mtm = !qa *. pa +. !qb *. pb in
          nav := prev +. (mtm -. (match !pnl with x::_ -> x | [] -> 0.));
          pnl := mtm :: (match !pnl with _::t -> t | [] -> []);
          let z = u.Kalman.standardised_innovation in
          if !pos = 0. && Float.abs z >= cfg.Config.entry_threshold then begin
            let dir = if z > 0. then -1. else 1. in
            qa := dir *. 25000. /. pa; qb := -. dir *. 25000. /. pb; pos := dir;
            pnl := (!qa *. pa +. !qb *. pb) :: (match !pnl with _::t->t|[]->[])
          end else if !pos <> 0. && Float.abs z <= cfg.Config.exit_threshold then begin
            qa := 0.; qb := 0.; pos := 0.;
            pnl := 0. :: (match !pnl with _::t->t|[]->[])
          end;
          rets := (!nav -. prev) /. prev :: !rets
        end) filt;
      let r = Array.of_list !rets in
      let m = Result.get_ok (Metrics.mean r) and sd = Result.get_ok (Metrics.stddev r) in
      let rf = Float.pow 1.04 (1./.252.) -. 1. in
      let sh = if sd < 1e-12 then 0. else (m -. rf) /. sd *. sqrt 252. in
      Printf.printf "%9.3f" sh) [2.;5.;12.];
    Printf.printf "\n") [10.; 20.; 30.]
