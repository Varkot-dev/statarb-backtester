(** The event loop.

    {1 Timing — the core of the no-lookahead guarantee}

    Each iteration processes one bar [t] in this fixed order:

    + {b Mark to market.} Value the position held {e into} bar [t] at bar [t]'s
      prices. This is the only way NAV changes other than costs.
    + {b Reconcile.} Assert the accounting identity for bar [t]. A failure here
      aborts the backtest; a book that does not balance must never produce a
      reported number.
    + {b Execute the intent decided at bar [t-1].} The signal computed with data
      through [t-1] fills at bar [t]'s price. This is next-bar execution.
    + {b Record} the audit row for bar [t].
    + {b Compute the signal at bar [t]} from a {!Causal.view} bounded at [t],
      and decide the intent to be executed at [t+1].

    Step 5 happens {e after} step 3 and its result is not used until the next
    iteration. That ordering is what makes same-bar execution impossible: the
    intent variable is written at the end of an iteration and read at the start
    of the next one, so there is no point in the loop where a decision could
    consume the price it fills at.

    Combined with the fact that step 5 can only see a bounded view, this gives
    the property [test/test_lookahead.ml] verifies: truncating the input after
    bar [t] leaves every signal at or before [t] bit-identical.

    {1 Why the spread history is grown incrementally}

    [spread_history] accumulates one spread per bar, each computed with the
    hedge ratio estimated at {e that} bar. The z-score at [t] standardises
    against this history. The alternative — recomputing all past spreads with
    today's beta — would rewrite history using future information. See
    {!Signal.compute}. *)

open Types

type backtest_result = {
  bars : bar_record list;  (** Chronological. *)
  trades : trade list;  (** Chronological. *)
  metrics : Metrics.t;
  navs : float array;
  config : Config.t;
}

(** [run cfg series] executes the full backtest.

    Returns [Error] on malformed configuration, insufficient data, or an
    accounting reconciliation failure. Warm-up conditions ([Insufficient_data]
    from a not-yet-full window) are expected and handled internally, not
    surfaced as errors. *)
let run (cfg : Config.t) (series : series) : (backtest_result, error) result =
  let open R in
  let n = Array.length series in
  let warmup = Config.warmup_bars cfg in
  if n < warmup + 2 then
    Error (Insufficient_data { needed = warmup + 2; got = n })
  else begin
    (* Log prices are precomputed once. Prices are validated positive at load
       time, so the log is always finite. *)
    let log_a = Array.map (fun b -> log (Price.to_float b.price_a)) series in
    let log_b = Array.map (fun b -> log (Price.to_float b.price_b)) series in

    (* Spread history grows one element per bar once the hedge window is full.
       Preallocated to n and tracked with an explicit count, so a view over it
       is bounded by [spread_count - 1], never by the array's capacity. *)
    let spread_history = Array.make n 0. in
    let spread_count = ref 0 in
    (* Bar index -> date, recorded at entry so a closed trade can report the
       date it was opened without the position carrying a string. *)
    let entry_dates : (int, string) Hashtbl.t = Hashtbl.create 64 in

    let portfolio = ref (Portfolio.initial cfg) in
    let pending_intent = ref Signal.Hold in
    let pending_beta = ref 0. in
    let pending_z = ref 0. in
    let records = ref [] in
    let navs = ref [] in
    let turnover_notional = ref 0. in
    let total_interest = ref 0. in
    let bars_with_position = ref 0 in
    let n_trading_bars = ref 0 in
    let abort : error option ref = ref None in

    let prev_nav = ref cfg.initial_capital in
    let prev_price_a = ref (Price.to_float series.(0).price_a) in
    let prev_price_b = ref (Price.to_float series.(0).price_b) in

    let i = ref 0 in
    while !i < n && !abort = None do
      let t = !i in
      let bar = series.(t) in
      let pa = Price.to_float bar.price_a in
      let pb = Price.to_float bar.price_b in

      (* --- Step 0: accrue one bar of interest on the cash carried into this
         bar. Bar 0 accrues nothing: no time has elapsed since the account was
         funded. --- *)
      let interest_this_bar =
        if t = 0 then 0.
        else begin
          let updated, interest = Portfolio.accrue_interest !portfolio cfg in
          portfolio := updated;
          total_interest := !total_interest +. interest;
          interest
        end
      in

      (* --- Step 1/2: the position held into this bar is marked at this bar's
         prices, and the accounting identity is checked. --- *)
      let position_into_bar = !portfolio.position in
      let nav_before_trades = Portfolio.nav !portfolio ~price_a:pa ~price_b:pb in

      (match
         Portfolio.check_reconciliation ~prev_nav:!prev_nav
           ~prev_position:position_into_bar ~prev_price_a:!prev_price_a
           ~prev_price_b:!prev_price_b ~price_a:pa ~price_b:pb
           ~new_nav:nav_before_trades ~costs_this_bar:0.
           ~interest_this_bar
       with
      | Ok () -> ()
      | Error msg ->
          abort :=
            Some
              (Config_error
                 (Printf.sprintf "bar %d (%s): %s" t bar.date msg)));

      if !abort = None then begin
        (* --- Step 3: execute the intent decided at bar t-1, at bar t's
           prices. --- *)
        let fill_result =
          match !pending_intent with
          | Signal.Hold -> Ok (Execution.no_fill !portfolio.position)
          | intent ->
              Execution.apply cfg ~intent ~position:!portfolio.position
                ~fill_price_a:pa ~fill_price_b:pb ~fill_index:t
                ~fill_date:bar.date ~beta:!pending_beta ~z:!pending_z
                ~entry_dates
        in
        match fill_result with
        | Error e -> abort := Some e
        | Ok fill ->
            (* Turnover: absolute notional traded, counted when a position is
               opened or closed. *)
            (if fill.event <> "" then
               let traded =
                 match (!portfolio.position, fill.new_position) with
                 | Flat, Open p | Open p, Flat ->
                     Execution.gross_notional p ~price_a:pa ~price_b:pb
                 | Open o, Open nw ->
                     Execution.gross_notional o ~price_a:pa ~price_b:pb
                     +. Execution.gross_notional nw ~price_a:pa ~price_b:pb
                 | Flat, Flat -> 0.
               in
               turnover_notional := !turnover_notional +. traded);

            portfolio := Portfolio.apply_fill !portfolio fill;
            let nav_after = Portfolio.nav !portfolio ~price_a:pa ~price_b:pb in

            (* The trade itself is NAV-neutral before costs; verify. *)
            let expected_after = nav_before_trades -. fill.cost in
            if Float.abs (expected_after -. nav_after)
               > Portfolio.reconciliation_tolerance
            then
              abort :=
                Some
                  (Config_error
                     (Printf.sprintf
                        "bar %d (%s): trade is not NAV-neutral before costs: \
                         before %.10f, cost %.10f, after %.10f"
                        t bar.date nav_before_trades fill.cost nav_after));

            if !abort = None then begin
              (* --- Step 5: compute this bar's signal from a bounded view. --- *)
              let va = Causal.create log_a t and vb = Causal.create log_b t in
              let snap_opt =
                if t < cfg.hedge_window - 1 then None
                else
                  let sv =
                    if !spread_count = 0 then None
                    else Some (Causal.create spread_history (!spread_count - 1))
                  in
                  match Signal.compute ~log_a:va ~log_b:vb ~spread_view:sv cfg with
                  | Ok s -> Some s
                  | Error (Insufficient_data _) | Error (Degenerate_regression _)
                    ->
                      (* Expected during warm-up or on a degenerate window.
                         Not an error: emit no signal this bar. *)
                      None
                  | Error e ->
                      abort := Some e;
                      None
              in

              (* Append this bar's spread to the history, using this bar's
                 hedge ratio. *)
              (match snap_opt with
              | Some s ->
                  spread_history.(!spread_count) <- s.spread;
                  incr spread_count
              | None -> ());

              (* Decide the intent to be executed at bar t+1. *)
              (match snap_opt with
              | Some s ->
                  pending_intent :=
                    Signal.decide ~snapshot:s ~position:!portfolio.position
                      ~bar_index:t cfg;
                  pending_beta := s.fit.beta;
                  pending_z := (match s.z with Some z -> z | None -> 0.)
              | None -> pending_intent := Signal.Hold);

              (* --- Step 4: record the audit row. --- *)
              let qa, qb =
                match !portfolio.position with
                | Flat -> (0., 0.)
                | Open p -> (signed_qty p Leg_a, signed_qty p Leg_b)
              in
              let rec_row =
                {
                  r_index = t;
                  r_date = bar.date;
                  r_price_a = pa;
                  r_price_b = pb;
                  r_hedge_ratio =
                    (match snap_opt with Some s -> Some s.fit.beta | None -> None);
                  r_spread =
                    (match snap_opt with Some s -> Some s.spread | None -> None);
                  r_zscore = (match snap_opt with Some s -> s.z | None -> None);
                  r_position =
                    (match !portfolio.position with
                    | Flat -> "flat"
                    | Open p -> string_of_direction p.dir);
                  r_qty_a = qa;
                  r_qty_b = qb;
                  r_cash = !portfolio.cash;
                  r_position_value =
                    Portfolio.position_value !portfolio ~price_a:pa ~price_b:pb;
                  r_nav = nav_after;
                  r_costs_this_bar = fill.cost;
                  r_interest_this_bar = interest_this_bar;
                  r_trade_event = fill.event;
                }
              in
              records := rec_row :: !records;
              navs := nav_after :: !navs;
              if t >= warmup then incr n_trading_bars;
              if not (is_flat !portfolio.position) && t >= warmup then
                incr bars_with_position;

              prev_nav := nav_after;
              prev_price_a := pa;
              prev_price_b := pb
            end
      end;
      incr i
    done;

    match !abort with
    | Some e -> Error e
    | None ->
        (* Close any position still open at the end of the sample, marked at
           the final bar's prices. Leaving it open would let an unrealised
           loss escape the trade statistics while still being reflected in
           NAV — an inconsistency between the two reported views. *)
        let final = series.(n - 1) in
        let fpa = Price.to_float final.price_a in
        let fpb = Price.to_float final.price_b in
        (* The liquidation must update EVERY reported view, not just NAV.

           A previous version patched [navs] alone and left [records],
           [turnover_notional], and [bars_with_position] untouched. The result
           shipped in published output: the final bar of bars.csv reported an
           open position with zero cost while metrics.csv reported it
           liquidated, the two disagreed on final NAV by the exit cost, and
           sum(bars.costs_this_bar) no longer equalled metrics.total_costs.

           That is the worst possible bug for this project, because the audit
           trail is precisely what the README asks readers to trust INSTEAD of
           the engine's own summary. Every view is now updated together. *)
        let* portfolio_final, liquidation =
          match !portfolio.position with
          | Flat -> Ok (!portfolio, None)
          | Open p ->
              let traded = Execution.gross_notional p ~price_a:fpa ~price_b:fpb in
              let* fill =
                Execution.apply cfg ~intent:(Signal.Exit End_of_data)
                  ~position:!portfolio.position ~fill_price_a:fpa
                  ~fill_price_b:fpb ~fill_index:(n - 1) ~fill_date:final.date
                  ~beta:!pending_beta ~z:!pending_z ~entry_dates
              in
              turnover_notional := !turnover_notional +. traded;
              (* The bar was counted as holding a position while the loop ran;
                 it no longer does, so the exposure numerator is corrected. *)
              if n - 1 >= warmup && !bars_with_position > 0 then
                decr bars_with_position;
              Ok (Portfolio.apply_fill !portfolio fill, Some fill)
        in
        let final_nav =
          Portfolio.nav portfolio_final ~price_a:fpa ~price_b:fpb
        in
        let navs_arr =
          let a = Array.of_list (List.rev !navs) in
          if Array.length a > 0 then a.(Array.length a - 1) <- final_nav;
          a
        in
        (* Rewrite the final audit row to match the liquidated book. Without
           this the row contradicts both metrics.csv and trades.csv. *)
        let recs =
          match (List.rev !records, liquidation) with
          | [], _ | _, None -> List.rev !records
          | rows, Some fill ->
              let last = List.nth rows (List.length rows - 1) in
              let patched =
                {
                  last with
                  r_position = "flat";
                  r_qty_a = 0.;
                  r_qty_b = 0.;
                  r_cash = portfolio_final.cash;
                  r_position_value = 0.;
                  r_nav = final_nav;
                  r_costs_this_bar = last.r_costs_this_bar +. fill.cost;
                  r_trade_event =
                    (if last.r_trade_event = "" then fill.event
                     else last.r_trade_event ^ ";" ^ fill.event);
                }
              in
              List.mapi
                (fun i r -> if i = List.length rows - 1 then patched else r)
                rows
        in
        let* metrics =
          Metrics.compute ~navs:navs_arr
            ~trades:(List.rev portfolio_final.trades)
            ~total_costs:portfolio_final.total_costs
            ~turnover_notional:!turnover_notional
            ~total_interest:!total_interest ~n_trading_bars:!n_trading_bars
            ~bars_with_position:!bars_with_position cfg
        in
        Ok
          {
            bars = recs;
            trades = List.rev portfolio_final.trades;
            metrics;
            navs = navs_arr;
            config = cfg;
          }
  end

(** [signals_only cfg series] runs the signal pipeline without any portfolio
    accounting, returning [(hedge_ratio, spread, zscore) option] per bar.

    This exists for the lookahead test: it isolates the signal path so the test
    can assert that signals through bar [t] are unchanged when all data after
    [t] is deleted, without the portfolio state making the comparison
    indirect. It uses exactly the same code path as {!run} — same
    {!Signal.compute}, same incremental spread history — so a bug that affected
    one would affect the other. *)
let signals_only (cfg : Config.t) (series : series) :
    ((float * float * float option) option array, error) result =
  let n = Array.length series in
  if n = 0 then Error (Insufficient_data { needed = 1; got = 0 })
  else begin
    let log_a = Array.map (fun b -> log (Price.to_float b.price_a)) series in
    let log_b = Array.map (fun b -> log (Price.to_float b.price_b)) series in
    let spread_history = Array.make n 0. in
    let spread_count = ref 0 in
    let out = Array.make n None in
    let err = ref None in
    let t = ref 0 in
    while !t < n && !err = None do
      let i = !t in
      if i >= cfg.hedge_window - 1 then begin
        let va = Causal.create log_a i and vb = Causal.create log_b i in
        let sv =
          if !spread_count = 0 then None
          else Some (Causal.create spread_history (!spread_count - 1))
        in
        match Signal.compute ~log_a:va ~log_b:vb ~spread_view:sv cfg with
        | Ok s ->
            spread_history.(!spread_count) <- s.spread;
            incr spread_count;
            out.(i) <- Some (s.fit.beta, s.spread, s.z)
        | Error (Insufficient_data _) | Error (Degenerate_regression _) -> ()
        | Error e -> err := Some e
      end;
      incr t
    done;
    match !err with Some e -> Error e | None -> Ok out
  end
