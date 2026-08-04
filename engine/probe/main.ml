open Statarb
open Statarb.Types

let mk_pos ~dir ~qa ~qb ~pa ~pb ~idx : open_position =
  {
    dir;
    qty_a = Fx.get_ok (Qty.of_float qa);
    qty_b = Fx.get_ok (Qty.of_float qb);
    entry_price_a = Fx.get_ok (Price.of_float pa);
    entry_price_b = Fx.get_ok (Price.of_float pb);
    entry_index = idx;
    entry_hedge_ratio = 1.0;
    entry_z = -2.5;
  }

(* Find a (seed, n) such that the backtest ends with a position still OPEN,
   which forces the end-of-data liquidation path. *)
let find_open_at_end cfg =
  let found = ref None in
  let seed = ref 1 in
  while !found = None && !seed < 400 do
    let n = 300 in
    let series =
      Fx.cointegrated ~n ~seed:!seed ~beta:1.15 ~half_life:12.
        ~sigma_spread:0.03 ~sigma_common:0.015
    in
    (match Backtest.run cfg series with
    | Ok res ->
        let bars = Array.of_list res.bars in
        let last = bars.(Array.length bars - 1) in
        (* end_of_data trade present => a position was open at the end *)
        if res.metrics.n_end_of_data > 0 then found := Some (!seed, series, res, last)
    | Error _ -> ());
    incr seed
  done;
  !found

(* ================================================================== *)
(* ATTACK 1: NAV RECONCILIATION / END-OF-DATA LIQUIDATION             *)
(* ================================================================== *)

let attack_eod () =
  Fx.hdr "ATTACK 1: forced end-of-data liquidation vs the audit trail";
  let cfg = Fx.test_config ~commission_bps:5. ~slippage_bps:5. () in
  match find_open_at_end cfg with
  | None -> Printf.printf "  (could not find a seed ending with an open position)\n"
  | Some (seed, series, res, last) ->
      Printf.printf "  seed=%d n_end_of_data=%d n_trades=%d\n" seed
        res.metrics.n_end_of_data res.metrics.n_trades;
      let bars = Array.of_list res.bars in
      let nb = Array.length bars in
      Printf.printf
        "  LAST BAR RECORD: idx=%d pos=%S qty_a=%.4f qty_b=%.4f\n\
        \                   cash=%.6f pos_val=%.6f nav=%.6f costs_this_bar=%.6f event=%S\n"
        last.r_index last.r_position last.r_qty_a last.r_qty_b last.r_cash
        last.r_position_value last.r_nav last.r_costs_this_bar last.r_trade_event;
      Printf.printf "  metrics.final_nav = %.6f\n" res.metrics.final_nav;
      Printf.printf "  navs.(last)       = %.6f\n" res.navs.(nb - 1);

      (* --- 1a: does the last bar record agree with the reported final NAV? --- *)
      Fx.ck "1a: last bar record's r_nav equals metrics.final_nav"
        (Float.abs (last.r_nav -. res.metrics.final_nav) < 1e-6);

      (* --- 1b: does the last bar record still show an OPEN position, while
         the reported book was liquidated? --- *)
      Fx.ck "1b: last bar record shows flat (book was liquidated)"
        (last.r_position = "flat");

      (* --- 1c: NAV = cash + position_value on the LAST bar record --- *)
      Fx.ckf "1c: last bar: NAV = cash + position value"
        (last.r_cash +. last.r_position_value)
        last.r_nav 1e-6;

      (* --- 1d: sum of r_costs_this_bar over bars.csv vs metrics.total_costs --- *)
      let summed_costs =
        List.fold_left (fun a r -> a +. r.r_costs_this_bar) 0. res.bars
      in
      Printf.printf "  sum(bars.r_costs_this_bar) = %.6f\n" summed_costs;
      Printf.printf "  metrics.total_costs        = %.6f\n" res.metrics.total_costs;
      Fx.ckf "1d: bars.csv costs sum to metrics.total_costs" res.metrics.total_costs
        summed_costs 1e-6;

      (* --- 1e: sum of trade.costs vs metrics.total_costs --- *)
      let trade_costs =
        List.fold_left (fun a (t : trade) -> a +. t.costs) 0. res.trades
      in
      Printf.printf "  sum(trades.costs)          = %.6f\n" trade_costs;
      Fx.ckf "1e: sum of trade costs = metrics.total_costs" res.metrics.total_costs
        trade_costs 1e-6;

      (* --- 1f: the external NAV-evolution identity, applied to the LAST bar,
         using the audit trail only (this is exactly what test_execution.ml's
         test_nav_reconciles_every_bar does, but that test never hits the
         liquidation because it uses navs from bar records, not navs_arr). --- *)
      Fx.hdr "ATTACK 1f: NAV-evolution identity over navs_arr (the reported series)";
      let navs = res.navs in
      let bad = ref 0 in
      for i = 1 to Array.length navs - 1 do
        let prev = bars.(i - 1) and cur = bars.(i) in
        let mtm =
          (prev.r_qty_a *. (cur.r_price_a -. prev.r_price_a))
          +. (prev.r_qty_b *. (cur.r_price_b -. prev.r_price_b))
        in
        let expected =
          navs.(i - 1) +. mtm +. cur.r_interest_this_bar -. cur.r_costs_this_bar
        in
        if Float.abs (expected -. navs.(i)) > 1e-6 then begin
          incr bad;
          if !bad <= 3 then
            Printf.printf
              "  BREAK at i=%d (%s): expected %.6f got %.6f diff %.6f\n" i
              cur.r_date expected navs.(i)
              (navs.(i) -. expected)
        end
      done;
      Fx.ck "1f: the reported NAV series satisfies the reconciliation identity"
        (!bad = 0);

      (* --- 1g: turnover. The liquidation trades real notional but is it
         counted? --- *)
      Fx.hdr "ATTACK 1g: turnover excludes the forced liquidation";
      let liq_notional =
        (Float.abs last.r_qty_a *. last.r_price_a)
        +. (Float.abs last.r_qty_b *. last.r_price_b)
      in
      Printf.printf "  reported turnover = %.6f  (x capital = %.2f notional)\n"
        res.metrics.turnover
        (res.metrics.turnover *. cfg.initial_capital);
      Printf.printf "  liquidation notional at last bar = %.2f\n" liq_notional;
      (* Reconstruct turnover from the audit trail: every bar with a trade
         event contributes its traded gross notional. *)
      let recon_turnover = ref 0. in
      Array.iteri
        (fun i r ->
          if r.r_trade_event <> "" then begin
            (* entry: notional of the new position at this bar
               exit: notional of the position held into this bar *)
            if String.length r.r_trade_event >= 5
               && String.sub r.r_trade_event 0 5 = "entry"
            then
              recon_turnover :=
                !recon_turnover
                +. (Float.abs r.r_qty_a *. r.r_price_a)
                +. (Float.abs r.r_qty_b *. r.r_price_b)
            else if String.length r.r_trade_event >= 4
                    && String.sub r.r_trade_event 0 4 = "exit"
            then
              let prev = bars.(i - 1) in
              recon_turnover :=
                !recon_turnover
                +. (Float.abs prev.r_qty_a *. r.r_price_a)
                +. (Float.abs prev.r_qty_b *. r.r_price_b)
          end)
        bars;
      Printf.printf "  turnover reconstructed from bars.csv = %.2f\n"
        !recon_turnover;
      Fx.ckf "1g: reported turnover notional = audit-trail turnover + liquidation"
        (!recon_turnover +. liq_notional)
        (res.metrics.turnover *. cfg.initial_capital)
        1.0;

      (* --- 1h: exposure / bars_with_position on the final bar --- *)
      ignore series

let () =
  attack_eod ();
  Printf.printf "\n=== probe failures: %d ===\n" !Fx.n_fail
