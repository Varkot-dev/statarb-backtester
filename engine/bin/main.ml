(** Command-line entry point for the backtest engine.

    Two subcommands:

    - [backtest]: run a single backtest and emit bars, trades, and metrics.
    - [sweep]: run a grid of parameter combinations and emit one metrics row
      per combination, for the sensitivity analysis in the README.

    Argument parsing is hand-rolled. The interface is small and fully under our
    control, and a dependency on [cmdliner] would add a package for a dozen
    flags. Unknown flags are a hard error rather than being ignored, so a typo
    in the Makefile cannot silently run a backtest with different parameters
    than the one the README claims to report. *)

open Statarb

let usage =
  {|statarb — pairs-trading backtest engine

USAGE
  statarb backtest --prices <in.csv> --out-dir <dir> [options]
  statarb sweep    --prices <in.csv> --out <sweep.csv> [options]
  statarb calibrate --prices <in.csv> --out <calibration.csv> [options]

CALIBRATE
  Measures how much a KNOWN dose of lookahead bias would change the reported
  result, producing a dose-response curve. Two leak types are dosed:

    timing shift    the signal at bar t is the one belonging to bar t+k.
                    Caused by an off-by-one in a resample or join.
    outcome filter  a fraction of trades that will lose are skipped.
                    Caused by shift(-1) on a feature, or same-bar execution.

  Dose zero reproduces the honest engine exactly. See lib/leakage.ml.

REQUIRED
  --prices PATH        input CSV: date,price_a,price_b (chronological)
  --out-dir DIR        output directory (backtest); writes bars.csv,
                       trades.csv, metrics.csv
  --out PATH           output CSV (sweep)

STRATEGY OPTIONS (defaults in parentheses)
  --hedge-window N     rolling OLS window for the hedge ratio (60)
  --zscore-window N    rolling window for the spread z-score (60)
  --entry FLOAT        entry threshold in sigmas (2.0)
  --exit FLOAT         exit threshold in sigmas (0.5)
  --stop FLOAT         stop-loss threshold in sigmas (3.5)
  --max-hold N         maximum holding period in bars (60)

COST AND SIZING OPTIONS
  --commission-bps F   per-leg, per-side commission (1.0)
  --slippage-bps F     per-leg, per-side slippage (2.0)
  --capital F          initial capital (100000.0)
  --frac F             fraction of initial capital per trade, leg A (0.25)

METRIC CONVENTIONS
  --bars-per-year F    annualization factor (252.0)
  --risk-free F        annual risk-free rate as a decimal (0.04)
  --no-cash-interest   do NOT credit idle cash at the risk-free rate.
                       Off by default: a part-time strategy that is charged a
                       risk-free hurdle on its whole NAV but earns nothing on
                       idle cash reports a Sharpe that mostly measures how
                       often it was flat. See Config.accrue_cash_interest.

OTHER
  --label STRING       label recorded in sweep output rows
  --quiet              suppress the human-readable summary
  -h, --help           show this message
|}

type opts = {
  mutable prices : string option;
  mutable out_dir : string option;
  mutable out : string option;
  mutable hedge_window : int;
  mutable zscore_window : int;
  mutable entry : float;
  mutable exit_ : float;
  mutable stop : float;
  mutable max_hold : int;
  mutable commission_bps : float;
  mutable slippage_bps : float;
  mutable capital : float;
  mutable frac : float;
  mutable bars_per_year : float;
  mutable risk_free : float;
  mutable label : string;
  mutable no_cash_interest : bool;
  mutable quiet : bool;
}

let default_opts () =
  let d = Config.default in
  {
    prices = None;
    out_dir = None;
    out = None;
    hedge_window = d.hedge_window;
    zscore_window = d.zscore_window;
    entry = d.entry_threshold;
    exit_ = d.exit_threshold;
    stop = d.stop_loss_threshold;
    max_hold = d.max_holding_bars;
    commission_bps = d.commission_bps;
    slippage_bps = d.slippage_bps;
    capital = d.initial_capital;
    frac = d.capital_per_trade_frac;
    bars_per_year = d.bars_per_year;
    risk_free = d.risk_free_rate;
    label = "";
    no_cash_interest = false;
    quiet = false;
  }

exception Bad_usage of string

let parse_args (argv : string array) (start : int) : opts =
  let o = default_opts () in
  let n = Array.length argv in
  let need i flag =
    if i + 1 >= n then raise (Bad_usage (flag ^ " requires a value"));
    argv.(i + 1)
  in
  let int_of flag s =
    match int_of_string_opt s with
    | Some v -> v
    | None -> raise (Bad_usage (Printf.sprintf "%s: '%s' is not an integer" flag s))
  in
  let float_of flag s =
    match float_of_string_opt s with
    | Some v -> v
    | None -> raise (Bad_usage (Printf.sprintf "%s: '%s' is not a number" flag s))
  in
  let i = ref start in
  while !i < n do
    let a = argv.(!i) in
    let consumed =
      match a with
      | "--prices" -> o.prices <- Some (need !i a); 2
      | "--out-dir" -> o.out_dir <- Some (need !i a); 2
      | "--out" -> o.out <- Some (need !i a); 2
      | "--hedge-window" -> o.hedge_window <- int_of a (need !i a); 2
      | "--zscore-window" -> o.zscore_window <- int_of a (need !i a); 2
      | "--entry" -> o.entry <- float_of a (need !i a); 2
      | "--exit" -> o.exit_ <- float_of a (need !i a); 2
      | "--stop" -> o.stop <- float_of a (need !i a); 2
      | "--max-hold" -> o.max_hold <- int_of a (need !i a); 2
      | "--commission-bps" -> o.commission_bps <- float_of a (need !i a); 2
      | "--slippage-bps" -> o.slippage_bps <- float_of a (need !i a); 2
      | "--capital" -> o.capital <- float_of a (need !i a); 2
      | "--frac" -> o.frac <- float_of a (need !i a); 2
      | "--bars-per-year" -> o.bars_per_year <- float_of a (need !i a); 2
      | "--risk-free" -> o.risk_free <- float_of a (need !i a); 2
      | "--label" -> o.label <- need !i a; 2
      | "--no-cash-interest" -> o.no_cash_interest <- true; 1
      | "--quiet" -> o.quiet <- true; 1
      | "-h" | "--help" -> print_string usage; exit 0
      | other -> raise (Bad_usage ("unknown flag: " ^ other))
    in
    i := !i + consumed
  done;
  o

let config_of_opts (o : opts) : Config.t =
  match
    Config.create ~hedge_window:o.hedge_window ~zscore_window:o.zscore_window
      ~entry_threshold:o.entry ~exit_threshold:o.exit_
      ~stop_loss_threshold:o.stop ~max_holding_bars:o.max_hold
      ~commission_bps:o.commission_bps ~slippage_bps:o.slippage_bps
      ~initial_capital:o.capital ~capital_per_trade_frac:o.frac
      ~bars_per_year:o.bars_per_year ~risk_free_rate:o.risk_free
      ~accrue_cash_interest:(not o.no_cash_interest) ()
  with
  | Ok c -> c
  | Error e -> raise (Bad_usage (Types.string_of_error e))

let die msg =
  prerr_endline ("error: " ^ msg);
  exit 1

let require what = function Some x -> x | None -> die (what ^ " is required")

(** Human-readable summary. Deliberately reports the numbers as measured,
    including negative ones — the CSV is the source of truth and this is a
    convenience view of the same values. *)
let print_summary (m : Metrics.t) (cfg : Config.t) =
  let p fmt = Printf.printf fmt in
  p "\n=== Backtest summary ===\n";
  p "config: %s\n\n" (Config.to_string cfg);
  p "  bars                    %d (%d after warm-up)\n" m.n_bars m.n_trading_bars;
  p "  initial NAV             %.2f\n" m.initial_nav;
  p "  final NAV               %.2f\n" m.final_nav;
  p "  total return            %+.4f%%\n" (m.total_return *. 100.);
  p "  annualized return       %+.4f%%\n" (m.annualized_return *. 100.);
  p "  annualized volatility   %.4f%%\n" (m.annualized_volatility *. 100.);
  p "  Sharpe ratio            %+.4f   (rf=%.2f%%, x sqrt(%.0f))\n"
    m.sharpe_ratio (cfg.risk_free_rate *. 100.) cfg.bars_per_year;
  p "  Sharpe per bar          %+.6f\n" m.sharpe_per_bar;
  p "  max drawdown            %.4f%% (%d bars peak-to-recovery)\n"
    (m.max_drawdown *. 100.) m.max_drawdown_duration;
  p "  Calmar ratio            %+.4f\n" m.calmar_ratio;
  p "\n  trades                  %d (%d wins, %d losses)\n" m.n_trades m.n_wins
    m.n_losses;
  p "  win rate                %.2f%%\n" (m.win_rate *. 100.);
  p "  avg holding period      %.2f bars\n" m.avg_holding_bars;
  p "  avg win / avg loss      %.2f / %.2f\n" m.avg_win m.avg_loss;
  p "  profit factor           %.4f\n" m.profit_factor;
  p "  exit reasons            reversion=%d stop=%d max_hold=%d eod=%d\n"
    m.n_reversion m.n_stop_loss m.n_max_holding m.n_end_of_data;
  p "\n  gross PnL               %+.2f\n" m.gross_pnl;
  p "  total costs             %.2f\n" m.total_costs;
  p "  net PnL                 %+.2f\n" m.net_pnl;
  p "  turnover                %.2fx initial capital\n" m.turnover;
  p "  time in market          %.2f%% of trading bars\n" (m.exposure_frac *. 100.);
  p "\n"

let cmd_backtest (o : opts) =
  let prices = require "--prices" o.prices in
  let out_dir = require "--out-dir" o.out_dir in
  let cfg = config_of_opts o in
  match Csv_io.read_prices prices with
  | Error e -> die (Printf.sprintf "reading %s: %s" prices (Types.string_of_error e))
  | Ok series -> (
      match Backtest.run cfg series with
      | Error e -> die ("backtest failed: " ^ Types.string_of_error e)
      | Ok res ->
          let path name = Filename.concat out_dir name in
          let check = function
            | Ok () -> ()
            | Error e -> die ("writing output: " ^ Types.string_of_error e)
          in
          check (Csv_io.write_bars (path "bars.csv") res.bars);
          check (Csv_io.write_trades (path "trades.csv") res.trades);
          check
            (Csv_io.write_metrics (path "metrics.csv")
               (Metrics.to_csv_rows res.metrics));
          if not o.quiet then print_summary res.metrics cfg;
          Printf.printf "wrote %s, %s, %s\n" (path "bars.csv")
            (path "trades.csv") (path "metrics.csv"))

(** Parameter grid for the sensitivity analysis.

    The point of the sweep is to show how much the headline result depends on
    the specific parameter choice. A strategy whose Sharpe collapses outside a
    narrow parameter window is overfit; one that degrades smoothly is more
    likely to be measuring something real. The grid is fixed in code (not
    tuned) so the README's sensitivity table is reproducible. *)
let sweep_grid =
  let entries = [ 1.5; 2.0; 2.5; 3.0 ] in
  let exits = [ 0.0; 0.5; 1.0 ] in
  let windows = [ 30; 60; 90 ] in
  List.concat_map
    (fun w ->
      List.concat_map
        (fun e -> List.filter_map (fun x -> if x < e then Some (w, e, x) else None) exits)
        entries)
    windows

(** Emit the leakage dose-response curves.

    Both leak types are written to one file with a [leak_type] column, so the
    two curves can be compared directly — which is the point, since they move
    in opposite directions. *)
let cmd_calibrate (o : opts) =
  let prices = require "--prices" o.prices in
  let out = require "--out" o.out in
  let cfg = config_of_opts o in
  match Csv_io.read_prices prices with
  | Error e -> die (Printf.sprintf "reading %s: %s" prices (Types.string_of_error e))
  | Ok series -> (
      match
        ( Leakage.sweep cfg series ~max_peek:10,
          Leakage.sweep_outcome_filter cfg series ~steps:10 ~seed:42 )
      with
      | Error e, _ | _, Error e -> die ("calibration failed: " ^ Types.string_of_error e)
      | Ok shift_curve, Ok filter_curve ->
          let header =
            "leak_type,dose,sharpe_ratio,annualized_return,max_drawdown,n_trades,win_rate,final_nav"
          in
          let row leak dose (c : Leakage.calibration) =
            Printf.sprintf "%s,%.4f,%.6f,%.6f,%.6f,%d,%.6f,%.2f" leak dose
              c.sharpe c.annualized_return c.max_drawdown c.n_trades c.win_rate
              c.final_nav
          in
          let rows =
            List.map
              (fun (c : Leakage.calibration) ->
                row "timing_shift" (float_of_int c.peek_bars) c)
              shift_curve
            @ List.map (fun (f, c) -> row "outcome_filter" f c) filter_curve
          in
          (match Csv_io.write_lines out (header :: rows) with
          | Error e -> die ("writing calibration: " ^ Types.string_of_error e)
          | Ok () -> ());
          if not o.quiet then begin
            Printf.printf "\n=== Leakage calibration ===\n\n";
            Printf.printf "  Timing shift (signal describes bar t+k):\n";
            List.iter
              (fun (c : Leakage.calibration) ->
                Printf.printf "    k=%-2d  Sharpe %+.4f  trades %3d  win %.3f\n"
                  c.peek_bars c.sharpe c.n_trades c.win_rate)
              shift_curve;
            Printf.printf "\n  Outcome filter (fraction of losers skipped):\n";
            List.iter
              (fun (f, (c : Leakage.calibration)) ->
                Printf.printf
                  "    %3.0f%%  Sharpe %+.4f  trades %3d  win %.3f\n"
                  (f *. 100.) c.sharpe c.n_trades c.win_rate)
              filter_curve;
            Printf.printf "\n"
          end;
          Printf.printf "wrote %s (%d rows)\n" out (List.length rows))

let cmd_sweep (o : opts) =
  let prices = require "--prices" o.prices in
  let out = require "--out" o.out in
  match Csv_io.read_prices prices with
  | Error e -> die (Printf.sprintf "reading %s: %s" prices (Types.string_of_error e))
  | Ok series ->
      let header =
        "label,window,entry,exit,sharpe_ratio,annualized_return,max_drawdown,n_trades,win_rate,total_costs,final_nav"
      in
      let label = if o.label = "" then "default" else o.label in
      let rows =
        List.filter_map
          (fun (w, e, x) ->
            match
              Config.create ~hedge_window:w ~zscore_window:w ~entry_threshold:e
                ~exit_threshold:x ~stop_loss_threshold:(e +. 1.5)
                ~max_holding_bars:o.max_hold ~commission_bps:o.commission_bps
                ~slippage_bps:o.slippage_bps ~initial_capital:o.capital
                ~capital_per_trade_frac:o.frac ~bars_per_year:o.bars_per_year
                ~risk_free_rate:o.risk_free
                ~accrue_cash_interest:(not o.no_cash_interest) ()
            with
            | Error _ -> None
            | Ok cfg -> (
                match Backtest.run cfg series with
                | Error _ -> None
                | Ok res ->
                    let m = res.metrics in
                    Some
                      (Printf.sprintf
                         "%s,%d,%.2f,%.2f,%.6f,%.6f,%.6f,%d,%.6f,%.2f,%.2f"
                         label w e x m.sharpe_ratio m.annualized_return
                         m.max_drawdown m.n_trades m.win_rate m.total_costs
                         m.final_nav)))
          sweep_grid
      in
      (match Csv_io.write_lines out (header :: rows) with
      | Error e -> die ("writing sweep: " ^ Types.string_of_error e)
      | Ok () -> ());
      Printf.printf "wrote %s (%d parameter combinations)\n" out
        (List.length rows)

let () =
  let argv = Sys.argv in
  if Array.length argv < 2 then begin
    print_string usage;
    exit 1
  end;
  try
    match argv.(1) with
    | "backtest" -> cmd_backtest (parse_args argv 2)
    | "sweep" -> cmd_sweep (parse_args argv 2)
    | "calibrate" -> cmd_calibrate (parse_args argv 2)
    | "-h" | "--help" -> print_string usage
    | other ->
        prerr_endline ("error: unknown subcommand: " ^ other);
        print_string usage;
        exit 1
  with Bad_usage msg -> die msg
