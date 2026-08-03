(** Performance and risk metrics, implemented from first principles.

    Every metric here is computed from the NAV series produced by
    {!Portfolio}, so all of them are consistent with one another by
    construction. Each is unit-tested against hand-computed fixtures in
    [test/test_metrics.ml].

    {1 Stated conventions}

    A "Sharpe ratio" quoted without its conventions is not a well-defined
    number. This implementation uses:

    - {b Simple} (not log) returns, computed bar to bar from NAV.
    - {b Sample} standard deviation (denominator [n-1]).
    - The risk-free rate is {b de-annualized geometrically} to the bar
      frequency: [rf_bar = (1 + rf_annual)^(1/bars_per_year) - 1]. The common
      shortcut [rf_annual / bars_per_year] is a linear approximation; at 4% and
      252 bars the difference is ~0.3% of the daily rate, which is immaterial
      here but the geometric form is unambiguously correct and costs nothing.
    - Annualization by [sqrt(bars_per_year)]. This is the standard convention
      and assumes returns are serially uncorrelated. Pairs-trading returns are
      {e not} serially uncorrelated — a position is held across many bars, so
      returns cluster — which means this factor overstates the true annualized
      Sharpe. This is a known limitation of the convention, is noted in the
      README, and is why the raw per-bar Sharpe is also reported. *)

open Types

type t = {
  n_bars : int;
  n_trading_bars : int;  (** Bars after warm-up, when trading was possible. *)
  initial_nav : float;
  final_nav : float;
  total_return : float;  (** Simple, over the whole sample. *)
  annualized_return : float;  (** Geometric (CAGR-style). *)
  annualized_volatility : float;
  sharpe_ratio : float;  (** Annualized. *)
  sharpe_per_bar : float;  (** Un-annualized, for readers who distrust the factor. *)
  max_drawdown : float;  (** Negative or zero, as a fraction. *)
  max_drawdown_duration : int;  (** Bars from peak to recovery (or to end). *)
  calmar_ratio : float;
  n_trades : int;
  n_wins : int;
  n_losses : int;
  win_rate : float;
  avg_holding_bars : float;
  avg_win : float;
  avg_loss : float;
  profit_factor : float;
  total_costs : float;
  gross_pnl : float;
  net_pnl : float;
  turnover : float;  (** Total traded notional / initial capital. *)
  exposure_frac : float;  (** Fraction of trading bars with a position on. *)
  n_stop_loss : int;
  n_reversion : int;
  n_max_holding : int;
  n_end_of_data : int;
}

(** [mean xs] is the arithmetic mean. Errors on an empty array rather than
    returning nan. *)
let mean (xs : float array) : (float, error) result =
  let n = Array.length xs in
  if n = 0 then Error (Insufficient_data { needed = 1; got = 0 })
  else Ok (Array.fold_left ( +. ) 0. xs /. float_of_int n)

(** [stddev xs] is the sample standard deviation (denominator [n-1]).

    Two-pass, for the same numerical reason as {!Rolling.variance}. *)
let stddev (xs : float array) : (float, error) result =
  let n = Array.length xs in
  if n < 2 then Error (Insufficient_data { needed = 2; got = n })
  else
    match mean xs with
    | Error e -> Error e
    | Ok mu ->
        let ss = ref 0. in
        Array.iter
          (fun x ->
            let d = x -. mu in
            ss := !ss +. (d *. d))
          xs;
        Ok (sqrt (!ss /. float_of_int (n - 1)))

(** Volatility below which the Sharpe ratio is reported as 0 rather than
    dividing by ~0. A NAV series with zero variance has no risk-adjusted
    return to speak of; emitting [infinity] would poison the CSV and the
    README. *)
let min_volatility = 1e-12

(** [sharpe ~returns ~risk_free_annual ~bars_per_year] is the annualized Sharpe
    ratio.

    {[
      rf_bar  = (1 + rf_annual)^(1/bars_per_year) - 1
      excess  = returns - rf_bar
      sharpe  = mean(excess) / stddev(excess) * sqrt(bars_per_year)
    ]}

    Note the standard deviation is of {e excess} returns. Since [rf_bar] is a
    constant, this equals the standard deviation of raw returns; it is written
    this way because it is the definition, and because it stays correct if a
    time-varying risk-free series is ever substituted. *)
let sharpe ~(returns : float array) ~(risk_free_annual : float)
    ~(bars_per_year : float) : (float, error) result =
  if Array.length returns < 2 then
    Error (Insufficient_data { needed = 2; got = Array.length returns })
  else
    let rf_bar =
      Float.pow (1. +. risk_free_annual) (1. /. bars_per_year) -. 1.
    in
    let excess = Array.map (fun r -> r -. rf_bar) returns in
    match (mean excess, stddev excess) with
    | Error e, _ | _, Error e -> Error e
    | Ok mu, Ok sd ->
        if sd < min_volatility then Ok 0.
        else Ok (mu /. sd *. sqrt bars_per_year)

(** [sharpe_per_bar ~returns ~risk_free_annual ~bars_per_year] is the same
    quantity without the [sqrt(bars_per_year)] factor. *)
let sharpe_per_bar ~(returns : float array) ~(risk_free_annual : float)
    ~(bars_per_year : float) : (float, error) result =
  match sharpe ~returns ~risk_free_annual ~bars_per_year with
  | Error e -> Error e
  | Ok s -> Ok (s /. sqrt bars_per_year)

(** [max_drawdown navs] is the largest peak-to-trough decline as a {e negative}
    fraction, together with the duration in bars from the peak that preceded the
    worst trough until NAV recovers that peak (or the end of the sample).

    Algorithm: sweep once, tracking the running maximum. At each bar the
    drawdown is [nav / peak - 1], which is <= 0 by construction since
    [peak >= nav]. The minimum over the sweep is the max drawdown.

    Sign convention: negative. A max drawdown of -0.15 means a 15% decline.
    Returning a negative number (rather than the absolute value) means the
    property "max drawdown is always <= 0" is checkable, which the
    property-based tests do.

    Requires all NAVs strictly positive: with NAV <= 0 the account is wiped out
    and the ratio is meaningless. *)
let max_drawdown (navs : float array) : (float * int, error) result =
  let n = Array.length navs in
  if n = 0 then Error (Insufficient_data { needed = 1; got = 0 })
  else begin
    let bad = ref None in
    Array.iter
      (fun v -> if v <= 0. && !bad = None then bad := Some (Invalid_price v))
      navs;
    match !bad with
    | Some e -> Error e
    | None ->
        let peak = ref navs.(0) in
        let peak_idx = ref 0 in
        let worst = ref 0. in
        let worst_peak_idx = ref 0 in
        let worst_trough_idx = ref 0 in
        for i = 0 to n - 1 do
          if navs.(i) > !peak then begin
            peak := navs.(i);
            peak_idx := i
          end;
          let dd = (navs.(i) /. !peak) -. 1. in
          if dd < !worst then begin
            worst := dd;
            worst_peak_idx := !peak_idx;
            worst_trough_idx := i
          end
        done;
        (* Duration: from the peak preceding the worst trough until NAV first
           regains that peak. If it never recovers, measure to the end of the
           sample — an unrecovered drawdown is the more serious case and
           should not report a shorter duration than one that recovered. *)
        let peak_value = navs.(!worst_peak_idx) in
        let recovery = ref None in
        (try
           for i = !worst_trough_idx to n - 1 do
             if navs.(i) >= peak_value then begin
               recovery := Some i;
               raise Exit
             end
           done
         with Exit -> ());
        let end_idx = match !recovery with Some i -> i | None -> n - 1 in
        Ok (!worst, end_idx - !worst_peak_idx)
  end

(** [annualized_return ~initial ~final ~n_bars ~bars_per_year] is the geometric
    (CAGR) annualized return.

    {[ (final / initial)^(bars_per_year / n_bars) - 1 ]}

    Geometric rather than arithmetic, because the arithmetic mean of returns
    scaled up overstates what an investor actually earns whenever returns vary
    (the volatility drag). *)
let annualized_return ~(initial : float) ~(final : float) ~(n_bars : int)
    ~(bars_per_year : float) : (float, error) result =
  if initial <= 0. then Error (Invalid_price initial)
  else if final <= 0. then
    (* Total loss. CAGR is undefined (the root of a non-positive number); -100%
       is the honest report. *)
    Ok (-1.)
  else if n_bars < 1 then Error (Insufficient_data { needed = 1; got = n_bars })
  else
    let years = float_of_int n_bars /. bars_per_year in
    Ok (Float.pow (final /. initial) (1. /. years) -. 1.)

(** [calmar ~annualized_return ~max_drawdown] is annualized return divided by
    the absolute max drawdown. Zero when there was no drawdown, since an
    infinite Calmar is not informative. *)
let calmar ~(annualized_return : float) ~(max_drawdown : float) : float =
  let mdd = Float.abs max_drawdown in
  if mdd < min_volatility then 0. else annualized_return /. mdd

(** [compute ~navs ~trades ~total_costs ~turnover_notional ~n_trading_bars
      ~bars_with_position cfg] assembles the full metric set. *)
let compute ~(navs : float array) ~(trades : trade list) ~(total_costs : float)
    ~(turnover_notional : float) ~(n_trading_bars : int)
    ~(bars_with_position : int) (cfg : Config.t) : (t, error) result =
  let open R in
  let n = Array.length navs in
  if n < 2 then Error (Insufficient_data { needed = 2; got = n })
  else
    let* returns = Rolling.simple_returns navs in
    let initial_nav = navs.(0) and final_nav = navs.(n - 1) in
    let* sr = sharpe ~returns ~risk_free_annual:cfg.risk_free_rate
                ~bars_per_year:cfg.bars_per_year in
    let* srb = sharpe_per_bar ~returns ~risk_free_annual:cfg.risk_free_rate
                 ~bars_per_year:cfg.bars_per_year in
    let* mdd, mdd_dur = max_drawdown navs in
    let* ann_ret =
      annualized_return ~initial:initial_nav ~final:final_nav ~n_bars:(n - 1)
        ~bars_per_year:cfg.bars_per_year
    in
    let* vol = stddev returns in
    let ann_vol = vol *. sqrt cfg.bars_per_year in
    let n_trades = List.length trades in
    let wins = List.filter (fun t -> t.pnl_net > 0.) trades in
    let losses = List.filter (fun t -> t.pnl_net <= 0.) trades in
    let n_wins = List.length wins and n_losses = List.length losses in
    let sum_of f l = List.fold_left (fun a t -> a +. f t) 0. l in
    let gross_win = sum_of (fun t -> t.pnl_net) wins in
    let gross_loss = Float.abs (sum_of (fun t -> t.pnl_net) losses) in
    let count_reason r =
      List.length (List.filter (fun t -> t.reason = r) trades)
    in
    Ok
      {
        n_bars = n;
        n_trading_bars;
        initial_nav;
        final_nav;
        total_return = (final_nav /. initial_nav) -. 1.;
        annualized_return = ann_ret;
        annualized_volatility = ann_vol;
        sharpe_ratio = sr;
        sharpe_per_bar = srb;
        max_drawdown = mdd;
        max_drawdown_duration = mdd_dur;
        calmar_ratio = calmar ~annualized_return:ann_ret ~max_drawdown:mdd;
        n_trades;
        n_wins;
        n_losses;
        win_rate =
          (if n_trades = 0 then 0.
           else float_of_int n_wins /. float_of_int n_trades);
        avg_holding_bars =
          (if n_trades = 0 then 0.
           else
             sum_of (fun t -> float_of_int t.holding_bars) trades
             /. float_of_int n_trades);
        avg_win = (if n_wins = 0 then 0. else gross_win /. float_of_int n_wins);
        avg_loss =
          (if n_losses = 0 then 0. else -.gross_loss /. float_of_int n_losses);
        profit_factor =
          (if gross_loss < min_volatility then 0. else gross_win /. gross_loss);
        total_costs;
        gross_pnl = sum_of (fun t -> t.pnl_gross) trades;
        net_pnl = final_nav -. initial_nav;
        turnover = turnover_notional /. cfg.initial_capital;
        exposure_frac =
          (if n_trading_bars = 0 then 0.
           else float_of_int bars_with_position /. float_of_int n_trading_bars);
        n_stop_loss = count_reason Stop_loss;
        n_reversion = count_reason Reversion;
        n_max_holding = count_reason Max_holding;
        n_end_of_data = count_reason End_of_data;
      }

(** [to_csv_rows m] renders the metrics as [(name, value)] pairs for the
    machine-readable output the Python reporting layer and the README consume. *)
let to_csv_rows (m : t) : (string * string) list =
  let f name v = (name, Printf.sprintf "%.10f" v) in
  let i name v = (name, string_of_int v) in
  [
    i "n_bars" m.n_bars;
    i "n_trading_bars" m.n_trading_bars;
    f "initial_nav" m.initial_nav;
    f "final_nav" m.final_nav;
    f "total_return" m.total_return;
    f "annualized_return" m.annualized_return;
    f "annualized_volatility" m.annualized_volatility;
    f "sharpe_ratio" m.sharpe_ratio;
    f "sharpe_per_bar" m.sharpe_per_bar;
    f "max_drawdown" m.max_drawdown;
    i "max_drawdown_duration" m.max_drawdown_duration;
    f "calmar_ratio" m.calmar_ratio;
    i "n_trades" m.n_trades;
    i "n_wins" m.n_wins;
    i "n_losses" m.n_losses;
    f "win_rate" m.win_rate;
    f "avg_holding_bars" m.avg_holding_bars;
    f "avg_win" m.avg_win;
    f "avg_loss" m.avg_loss;
    f "profit_factor" m.profit_factor;
    f "total_costs" m.total_costs;
    f "gross_pnl" m.gross_pnl;
    f "net_pnl" m.net_pnl;
    f "turnover" m.turnover;
    f "exposure_frac" m.exposure_frac;
    i "n_stop_loss" m.n_stop_loss;
    i "n_reversion" m.n_reversion;
    i "n_max_holding" m.n_max_holding;
    i "n_end_of_data" m.n_end_of_data;
  ]
