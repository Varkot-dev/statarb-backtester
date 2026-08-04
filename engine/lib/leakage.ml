(** Controlled lookahead injection: a calibration instrument.

    {1 The idea}

    Every backtester claims to have no lookahead bias. The claim is almost never
    falsifiable, because a reader has no way to know what a {e given amount} of
    leakage would even look like in the reported numbers. "Sharpe 4.0 seems
    high" is a smell, not a measurement.

    This module makes it quantitative. It deliberately injects a measured dose
    of the exact bias the rest of the codebase is built to prevent, and runs the
    engine across a range of doses to produce {b dose-response curves}:

    {v
        amount of cheating  ->  the Sharpe ratio you would report
    v}

    The honest engine sits at dose zero. Everything else on the curve is what
    the same strategy, on the same data, would look like with a known quantity
    of contamination.

    {1 What the curves showed}

    The measured result (see the README) was not what the author predicted, and
    is the reason the instrument earns its place.

    {b Two different leaks push performance in opposite directions.}

    - {b Timing shift} — the signal at bar [t] is the one that belongs to
      [t + k]. Produced by an off-by-one in a resample, a join, or a
      [shift(-k)]. On the primary dataset this {e destroys} performance:
      Sharpe falls from +0.69 to −2.56 by [k = 6]. The reason is specific to
      mean reversion: the rule wants to enter {e at} an extreme, and foresight
      makes it enter early, so it holds the position while the spread travels
      {e into} the extreme.

    - {b Outcome filtering} — some part of the trade's eventual result bleeds
      into the decision to take it. Produced by a [shift(-1)] on a feature, by
      same-bar execution, or by dropping "bad" bars after seeing which ones
      hurt. This {e inflates} performance: Sharpe rises from +0.74 to +1.15 when
      every losing trade is skipped, while trade count falls and win rate
      climbs to 82%.

    {b The diagnostic value is the asymmetry.} These are different bugs with
    different signatures. A pairs backtest that suddenly reports a strongly
    negative Sharpe is more likely misaligned than broken-strategy. One that
    reports a suspiciously high Sharpe {e together with} an unusually high win
    rate and fewer trades than the signal count implies is showing the
    fingerprint of outcome contamination.

    {1 Why it is safe to keep leaky code in the repository}

    Leaky code next to honest code is a hazard, so the separation is structural
    rather than conventional:

    - This module is the {e only} place that can build a future-aware window,
      and it does so by taking a raw array rather than a {!Causal.view}. That
      asymmetry is the tell: anything taking a view is causal by construction;
      anything taking an array is under suspicion.
    - {!Backtest.run} — the production path — cannot call this module. It builds
      views and hands them to {!Signal.compute}.
    - [test_leakage.ml] asserts that a non-zero dose {e breaks} truncation
      invariance. If this were ever wired into the production path, the
      lookahead suite would fail immediately.
    - [test_zero_peek_matches_the_honest_engine] pins dose zero to the
      production engine bit-for-bit, so the curve's origin is the real
      backtest and not an approximation of it. *)

open Types

(** [spread_series cfg series] computes the honest, causal spread at every bar,
    returning [None] for bars before the hedge window is full.

    This is the shared input to both the honest and the leaky z-score, so the
    two differ {e only} in which window of this series they standardise
    against — not in how the spread itself was built. *)
let spread_series (cfg : Config.t) (series : series) :
    (float option array, error) result =
  let n = Array.length series in
  if n = 0 then Error (Insufficient_data { needed = 1; got = 0 })
  else begin
    let log_a = Array.map (fun b -> log (Price.to_float b.price_a)) series in
    let log_b = Array.map (fun b -> log (Price.to_float b.price_b)) series in
    let out = Array.make n None in
    let err = ref None in
    let t = ref 0 in
    while !t < n && !err = None do
      let i = !t in
      if i >= cfg.hedge_window - 1 then begin
        match
          Ols.fit_window ~y:(Causal.create log_a i) ~x:(Causal.create log_b i)
            cfg.hedge_window
        with
        | Ok fit ->
            out.(i) <-
              Some (Ols.residual fit ~y_t:log_a.(i) ~x_t:log_b.(i))
        | Error (Insufficient_data _) | Error (Degenerate_regression _) -> ()
        | Error e -> err := Some e
      end;
      incr t
    done;
    match !err with Some e -> Error e | None -> Ok out
  end

(** [leaky_zscores cfg series ~peek_bars] computes a z-score at each bar using
    a spread window that extends [peek_bars] into the future.

    [peek_bars = 0] reproduces the honest, causal calculation exactly — which
    is the property {!Test_leakage} asserts against the production engine.

    {2 Signal shifting, and why it is {e not} the dangerous leak}

    The obvious way to model lookahead is to shift the signal: at bar [t], use
    the z-score that will exist at [t + k]. That is what this function does, and
    measuring it produced a result worth recording, because it is the opposite
    of what one would guess.

    {b Shifting the z-score forward makes a mean-reversion strategy worse, not
    better.} The reason is specific to the entry rule. The strategy wants to
    enter {e at} an extreme and profit from the reversion. If at bar [t] it sees
    [z(t+3) = 2.5] while [z(t) = 0.1], it enters immediately — and then holds
    the position while the spread travels {e to} 2.5, losing the whole way, and
    only afterwards collects the reversion it was aiming for. Foresight makes it
    consistently early, and early is wrong for this rule.

    This is why the calibration curve is genuinely informative rather than
    decorative: a naive reader (and the author's first draft) would predict that
    any lookahead inflates Sharpe. For timing-shift leakage in a mean-reversion
    strategy, it does not. An off-by-one in a resample or a join produces this
    failure mode, and it shows up as {e underperformance}, which is a useful
    thing to know when debugging.

    The leak that {e does} inflate Sharpe is {!run_with_outcome_filter} — see
    there.

    The blend weight makes partial leakage expressible, since real bugs rarely
    leak a whole bar cleanly:

    {[ z_leaky(t) = (1 - w) * z_honest(t) + w * z_honest(t + k) ]}

    At [k = 0] this is exactly [z_honest(t)] for any [w], which anchors the
    curve to the production engine.

    {2 Matching the production window exactly}

    One further subtlety, easy to get wrong, that would invalidate the anchor.

    {!Backtest.run} computes the z-score at bar [t] {e before} appending bar
    [t]'s own spread, and {!Rolling.zscore} standardises [Causal.current] — the
    last element of the view. So the honest z at bar [t] compares {e
    yesterday's} spread against a window ending yesterday. Reproducing that
    exactly is what [test_zero_peek_matches_the_honest_engine] enforces.

    Note this deliberately takes the raw array of spreads, {e not} a
    {!Causal.view}. It could not be written against a view: [Causal.get] returns
    [None] past its bound. That is the whole point of the view type, and the
    signature of this function is what a violation of it looks like. *)
let leaky_zscores (cfg : Config.t) (spreads : float option array)
    ~(peek_bars : int) ?(weight = 1.0) () : float option array =
  let n = Array.length spreads in
  (* Compact the sparse spread array into a dense one, keeping the mapping from
     bar index to position, so windows are over consecutive real spreads. *)
  let dense = Array.make n 0. in
  let position_of_bar = Array.make n (-1) in
  let count = ref 0 in
  Array.iteri
    (fun i s ->
      match s with
      | Some v ->
          dense.(!count) <- v;
          position_of_bar.(i) <- !count;
          incr count
      | None -> ())
    spreads;

  (* [honest_z p] is the z-score production would compute for a bar at dense
     position [p]: the spread at [p - 1] standardised against the window
     ending at [p - 1]. See the note above on matching production exactly. *)
  let honest_z (p : int) : float option =
    let window_end = p - 1 in
    let window_start = window_end - cfg.zscore_window + 1 in
    if window_start < 0 || window_end >= !count then None
    else begin
      let k = window_end - window_start + 1 in
      let total = ref 0. in
      for j = window_start to window_end do
        total := !total +. dense.(j)
      done;
      let mean = !total /. float_of_int k in
      let ss = ref 0. in
      for j = window_start to window_end do
        let d = dense.(j) -. mean in
        ss := !ss +. (d *. d)
      done;
      let sd = sqrt (!ss /. float_of_int (k - 1)) in
      if sd < Rolling.min_stddev then None
      else Some ((dense.(window_end) -. mean) /. sd)
    end
  in

  Array.map
    (fun _ -> None)
    spreads
  |> fun out ->
  Array.iteri
    (fun i _ ->
      let p = position_of_bar.(i) in
      if p >= 0 then
        match honest_z p with
        | None -> ()
        | Some z_now ->
            if peek_bars = 0 || weight = 0. then out.(i) <- Some z_now
            else
              (* Blend in the z-score that will exist [peek_bars] later. Near
                 the end of the sample the future does not exist, so the honest
                 value is used — the alternative (dropping the bar) would
                 shorten the sample and make runs at different [k] cover
                 different periods, which would confound the comparison. *)
              let future = honest_z (p + peek_bars) in
              let z =
                match future with
                | Some z_future ->
                    ((1. -. weight) *. z_now) +. (weight *. z_future)
                | None -> z_now
              in
              out.(i) <- Some z)
    spreads;
  out

(** Result of one leakage-calibration run. *)
type calibration = {
  peek_bars : int;
  sharpe : float;
  annualized_return : float;
  max_drawdown : float;
  n_trades : int;
  win_rate : float;
  final_nav : float;
}

(** [run_with_leakage cfg series ~peek_bars] runs a full backtest whose signal
    peeks [peek_bars] bars ahead, and reports the metrics it would produce.

    The execution model is otherwise unchanged: still next-bar fills, still the
    same costs and sizing. Only the z-score is contaminated. This isolates the
    effect of foresight from every other modelling choice.

    At [peek_bars = 0] this must reproduce {!Backtest.run} exactly. *)
let run_with_leakage (cfg : Config.t) (series : series) ~(peek_bars : int)
    ?(weight = 1.0) () : (calibration, error) result =
  let open R in
  if peek_bars < 0 then
    Error (Config_error (Printf.sprintf "peek_bars must be >= 0, got %d" peek_bars))
  else
    let n = Array.length series in
    let warmup = Config.warmup_bars cfg in
    if n < warmup + 2 then
      Error (Insufficient_data { needed = warmup + 2; got = n })
    else
      let* spreads = spread_series cfg series in
      let zscores = leaky_zscores cfg spreads ~peek_bars ~weight () in

      (* A minimal replay of the event loop, driven by the precomputed z-scores.
         Timing is identical to Backtest.run: the intent decided at bar t is
         filled at bar t+1. *)
      let portfolio = ref (Portfolio.initial cfg) in
      let pending = ref Signal.Hold in
      let pending_z = ref 0. in
      let pending_beta = ref 1. in
      let entry_dates : (int, string) Hashtbl.t = Hashtbl.create 64 in
      let navs = ref [] in
      let abort = ref None in

      let i = ref 0 in
      while !i < n && !abort = None do
        let t = !i in
        let bar = series.(t) in
        let pa = Price.to_float bar.price_a in
        let pb = Price.to_float bar.price_b in

        if t > 0 then begin
          let updated, _ = Portfolio.accrue_interest !portfolio cfg in
          portfolio := updated
        end;

        (match !pending with
        | Signal.Hold -> ()
        | intent -> (
            match
              Execution.apply cfg ~intent ~position:!portfolio.position
                ~fill_price_a:pa ~fill_price_b:pb ~fill_index:t
                ~fill_date:bar.date ~beta:!pending_beta ~z:!pending_z
                ~entry_dates
            with
            | Ok fill -> portfolio := Portfolio.apply_fill !portfolio fill
            | Error e -> abort := Some e));

        if !abort = None then begin
          (match zscores.(t) with
          | Some z ->
              let snapshot =
                {
                  Signal.fit = { Ols.alpha = 0.; beta = 1.; r_squared = 0. };
                  spread = 0.;
                  z = Some z;
                }
              in
              pending :=
                Signal.decide ~snapshot ~position:!portfolio.position
                  ~bar_index:t cfg;
              pending_z := z
          | None -> pending := Signal.Hold);
          navs := Portfolio.nav !portfolio ~price_a:pa ~price_b:pb :: !navs
        end;
        incr i
      done;

      match !abort with
      | Some e -> Error e
      | None ->
          let final = series.(n - 1) in
          let fpa = Price.to_float final.price_a in
          let fpb = Price.to_float final.price_b in
          let* final_portfolio =
            match !portfolio.position with
            | Flat -> Ok !portfolio
            | Open _ ->
                let* fill =
                  Execution.apply cfg ~intent:(Signal.Exit End_of_data)
                    ~position:!portfolio.position ~fill_price_a:fpa
                    ~fill_price_b:fpb ~fill_index:(n - 1) ~fill_date:final.date
                    ~beta:!pending_beta ~z:!pending_z ~entry_dates
                in
                Ok (Portfolio.apply_fill !portfolio fill)
          in
          let final_nav =
            Portfolio.nav final_portfolio ~price_a:fpa ~price_b:fpb
          in
          let navs_arr =
            let a = Array.of_list (List.rev !navs) in
            if Array.length a > 0 then a.(Array.length a - 1) <- final_nav;
            a
          in
          let trades = List.rev final_portfolio.trades in
          let* metrics =
            Metrics.compute ~navs:navs_arr ~trades
              ~total_costs:final_portfolio.total_costs ~turnover_notional:0.
              ~total_interest:0. ~n_trading_bars:(n - warmup)
              ~bars_with_position:0 cfg
          in
          Ok
            {
              peek_bars;
              sharpe = metrics.sharpe_ratio;
              annualized_return = metrics.annualized_return;
              max_drawdown = metrics.max_drawdown;
              n_trades = metrics.n_trades;
              win_rate = metrics.win_rate;
              final_nav = metrics.final_nav;
            }

(** [run_with_outcome_filter cfg series ~fraction ~seed] runs the honest signal
    but declines to open a position when the trade is going to lose.

    {1 This is the leak that actually inflates Sharpe}

    Signal shifting (above) makes a mean-reversion strategy {e worse}. The bias
    that produces implausible backtest results is {b outcome filtering}: some
    part of the future outcome bleeds into the decision to take the trade at
    all, so the sample of trades is no longer the sample you could have taken.

    Real bugs that do this:

    - A [shift(-1)] on a feature column, so today's signal contains tomorrow's
      price and the model learns to avoid the bad days.
    - Same-bar execution: filling at a close you could not have observed before
      deciding to trade, which is a partial peek at the bar's outcome.
    - Dropping "bad data" or outlier bars {e after} seeing which ones hurt.
    - Selecting the strategy, the pair, or the parameters on the same sample
      that reports the result — the same bias one level up.

    Here it is dosed explicitly: with probability [fraction], a trade that will
    end up losing is skipped. [fraction = 0] is the honest engine;
    [fraction = 1] is perfect foresight, skipping every loser.

    {1 Why this needs a two-pass implementation}

    The first pass runs the honest backtest and records the outcome of every
    trade. The second pass replays it, skipping entries whose recorded outcome
    was a loss. Deciding to skip requires knowing the future, which is the whole
    point — and it is why this cannot be expressed against a {!Causal.view} and
    lives here rather than in {!Backtest}.

    The PRNG is a seeded LCG so the curve is reproducible; the exact constants
    do not matter, only that the same seed selects the same trades. *)
let run_with_outcome_filter (cfg : Config.t) (series : series)
    ~(fraction : float) ~(seed : int) : (calibration, error) result =
  let open R in
  if fraction < 0. || fraction > 1. then
    Error
      (Config_error
         (Printf.sprintf "fraction must be in [0, 1], got %g" fraction))
  else
    let* honest = Backtest.run cfg series in

    (* Which entry bars led to a losing trade? *)
    let losing_entry : (int, unit) Hashtbl.t = Hashtbl.create 64 in
    let state = ref (seed land 0x3FFFFFFF) in
    let next_uniform () =
      state := ((1664525 * !state) + 1013904223) land 0x3FFFFFFF;
      float_of_int !state /. 1073741824.
    in
    List.iter
      (fun (t : trade) ->
        if t.pnl_net <= 0. && next_uniform () < fraction then
          Hashtbl.replace losing_entry t.entry_index ())
      honest.trades;

    if Hashtbl.length losing_entry = 0 then
      (* Nothing was filtered, so the honest result stands. Reported through the
         same record type so the curve has a consistent shape. *)
      Ok
        {
          peek_bars = 0;
          sharpe = honest.metrics.sharpe_ratio;
          annualized_return = honest.metrics.annualized_return;
          max_drawdown = honest.metrics.max_drawdown;
          n_trades = honest.metrics.n_trades;
          win_rate = honest.metrics.win_rate;
          final_nav = honest.metrics.final_nav;
        }
    else begin
      (* Replay, skipping the flagged entries. The signal itself is untouched:
         only the decision to act on it is contaminated. *)
      let n = Array.length series in
      let bars = Array.of_list honest.bars in
      let portfolio = ref (Portfolio.initial cfg) in
      let entry_dates : (int, string) Hashtbl.t = Hashtbl.create 64 in
      let navs = ref [] in
      let pending = ref Signal.Hold in
      let pending_z = ref 0. in
      let pending_beta = ref 1. in
      let suppress_until_band_exit = ref false in
      let abort = ref None in

      let i = ref 0 in
      while !i < n && !abort = None do
        let t = !i in
        let bar = series.(t) in
        let pa = Price.to_float bar.price_a in
        let pb = Price.to_float bar.price_b in

        if t > 0 then begin
          let updated, _ = Portfolio.accrue_interest !portfolio cfg in
          portfolio := updated
        end;

        (* Skip the fill if this entry is one the cheat declined to take.

           Declining a single bar is not enough: the strategy would simply
           re-enter on the next bar that is still inside the entry band, so the
           losing trade would be delayed rather than avoided (measured: skipping
           100% of losers removed only 2 of 54 trades). Once an entry is
           declined, the whole episode is skipped — the engine stays flat until
           the z-score leaves the entry band, which is what "not taking this
           trade" actually means. *)
        let suppressed =
          match !pending with
          | Signal.Enter _ | Signal.Flip _ ->
              if Hashtbl.mem losing_entry t then begin
                suppress_until_band_exit := true;
                true
              end
              else !suppress_until_band_exit
          | _ -> false
        in
        (* The episode ends when the signal returns inside the entry band. *)
        (match bars.(t).r_zscore with
        | Some z when Float.abs z < cfg.entry_threshold ->
            suppress_until_band_exit := false
        | _ -> ());
        (match !pending with
        | Signal.Hold -> ()
        | _ when suppressed -> ()
        | intent -> (
            match
              Execution.apply cfg ~intent ~position:!portfolio.position
                ~fill_price_a:pa ~fill_price_b:pb ~fill_index:t
                ~fill_date:bar.date ~beta:!pending_beta ~z:!pending_z
                ~entry_dates
            with
            | Ok fill -> portfolio := Portfolio.apply_fill !portfolio fill
            | Error e -> abort := Some e));

        if !abort = None then begin
          (* Reuse the honest engine's z-scores: the signal is not the thing
             being corrupted here, the trade selection is. *)
          (match bars.(t).r_zscore with
          | Some z ->
              let snapshot =
                {
                  Signal.fit = { Ols.alpha = 0.; beta = 1.; r_squared = 0. };
                  spread = 0.;
                  z = Some z;
                }
              in
              pending :=
                Signal.decide ~snapshot ~position:!portfolio.position
                  ~bar_index:t cfg;
              pending_z := z
          | None -> pending := Signal.Hold);
          navs := Portfolio.nav !portfolio ~price_a:pa ~price_b:pb :: !navs
        end;
        incr i
      done;

      match !abort with
      | Some e -> Error e
      | None ->
          let final = series.(n - 1) in
          let fpa = Price.to_float final.price_a in
          let fpb = Price.to_float final.price_b in
          let* final_portfolio =
            match !portfolio.position with
            | Flat -> Ok !portfolio
            | Open _ ->
                let* fill =
                  Execution.apply cfg ~intent:(Signal.Exit End_of_data)
                    ~position:!portfolio.position ~fill_price_a:fpa
                    ~fill_price_b:fpb ~fill_index:(n - 1) ~fill_date:final.date
                    ~beta:!pending_beta ~z:!pending_z ~entry_dates
                in
                Ok (Portfolio.apply_fill !portfolio fill)
          in
          let final_nav =
            Portfolio.nav final_portfolio ~price_a:fpa ~price_b:fpb
          in
          let navs_arr =
            let a = Array.of_list (List.rev !navs) in
            if Array.length a > 0 then a.(Array.length a - 1) <- final_nav;
            a
          in
          let* metrics =
            Metrics.compute ~navs:navs_arr
              ~trades:(List.rev final_portfolio.trades)
              ~total_costs:final_portfolio.total_costs ~turnover_notional:0.
              ~total_interest:0.
              ~n_trading_bars:(n - Config.warmup_bars cfg)
              ~bars_with_position:0 cfg
          in
          Ok
            {
              peek_bars = 0;
              sharpe = metrics.sharpe_ratio;
              annualized_return = metrics.annualized_return;
              max_drawdown = metrics.max_drawdown;
              n_trades = metrics.n_trades;
              win_rate = metrics.win_rate;
              final_nav = metrics.final_nav;
            }
    end

(** [sweep_outcome_filter cfg series ~steps ~seed] produces the dose-response
    curve for outcome filtering: the fraction of losing trades avoided, against
    the Sharpe ratio that would be reported.

    This is the curve worth putting in a README. It answers: {e how much
    cheating would it take to produce the number I am looking at?} *)
let sweep_outcome_filter (cfg : Config.t) (series : series) ~(steps : int)
    ~(seed : int) : ((float * calibration) list, error) result =
  let rec go i acc =
    if i > steps then Ok (List.rev acc)
    else
      let fraction = float_of_int i /. float_of_int steps in
      match run_with_outcome_filter cfg series ~fraction ~seed with
      | Ok c -> go (i + 1) ((fraction, c) :: acc)
      | Error e -> Error e
  in
  go 0 []

(** [sweep cfg series ~max_peek] runs the calibration for every
    [peek_bars] in [0 .. max_peek], producing the dose-response curve. *)
let sweep (cfg : Config.t) (series : series) ~(max_peek : int) :
    (calibration list, error) result =
  let rec go k acc =
    if k > max_peek then Ok (List.rev acc)
    else
      match run_with_leakage cfg series ~peek_bars:k () with
      | Ok c -> go (k + 1) (c :: acc)
      | Error e -> Error e
  in
  go 0 []

let csv_header =
  "peek_bars,sharpe_ratio,annualized_return,max_drawdown,n_trades,win_rate,final_nav"

let csv_row (c : calibration) : string =
  Printf.sprintf "%d,%.6f,%.6f,%.6f,%d,%.6f,%.2f" c.peek_bars c.sharpe
    c.annualized_return c.max_drawdown c.n_trades c.win_rate c.final_nav
