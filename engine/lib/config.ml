(** Backtest configuration, with validation at construction.

    Every tunable lives here rather than as a literal in the logic. The
    validation in {!create} rules out combinations that are individually
    plausible but jointly incoherent — most importantly [entry <= exit], which
    would open and immediately close a position every bar and generate
    thousands of cost-only round trips. *)

open Types

type t = {
  (* --- Windows --- *)
  hedge_window : int;
      (** Trailing bars used to estimate the hedge ratio via rolling OLS. *)
  zscore_window : int;
      (** Trailing bars used for the mean/stddev of the spread. *)
  (* --- Thresholds, in units of standard deviations of the spread --- *)
  entry_threshold : float;
      (** Open when [|z| >= entry_threshold]. Convention: 2.0. *)
  exit_threshold : float;
      (** Close when [|z| <= exit_threshold]. Convention: 0.5. *)
  stop_loss_threshold : float;
      (** Close when [|z| >= stop_loss_threshold] against the position.
          Divergence past this point is treated as evidence the relationship
          has broken rather than an even better entry. *)
  max_holding_bars : int;
      (** Hard cap on holding period. A spread that has not reverted in this
          many bars is assumed non-stationary; without this cap a single broken
          pair can hold a losing position for the entire sample. *)
  (* --- Costs --- *)
  commission_bps : float;
      (** Per-side, per-leg commission in basis points of notional. *)
  slippage_bps : float;
      (** Per-side, per-leg slippage in basis points of notional. Models the
          gap between the reference price and the actual fill. *)
  (* --- Sizing --- *)
  initial_capital : float;
  capital_per_trade_frac : float;
      (** Fraction of {e initial} capital committed to leg A per trade. Using
          initial rather than current capital makes position size constant in
          dollar terms, which decouples the measured edge from the compounding
          path. See {!Execution} for the full sizing rule. *)
  (* --- Annualization --- *)
  bars_per_year : float;
      (** 252 for daily bars. Used for annualized return and Sharpe. *)
  risk_free_rate : float;
      (** Annual, as a decimal (0.04 = 4%). Stated explicitly because a "Sharpe
          ratio" quoted without its risk-free assumption is not a well-defined
          number.

          Used for two distinct purposes, which must be kept consistent:
          - as the hurdle subtracted from returns when computing Sharpe;
          - as the rate at which idle cash accrues, when {!accrue_cash_interest}
            is set. See that field for why the two must agree. *)
  accrue_cash_interest : bool;
      (** Whether uninvested cash earns {!risk_free_rate}.

          This exists because leaving it out produces a specific, subtle
          distortion in the Sharpe ratio.

          A pairs strategy is in the market only part of the time — here about
          40% of bars — and commits only a fraction of capital when it is. The
          rest sits in cash. If that cash earns nothing while the Sharpe
          calculation still charges the {e entire} NAV a risk-free hurdle, the
          strategy is penalised for holding cash but never credited for it. The
          resulting Sharpe is not a measure of the strategy's edge; it is
          mostly a measure of how often the strategy was flat.

          With accrual on, the comparison is coherent: the portfolio genuinely
          earns the risk-free rate on idle balances, exactly as a real margin
          account does, and the Sharpe ratio measures what it is supposed to —
          return in excess of the risk-free alternative.

          Defaults to [true]. Setting it to [false] with a non-zero
          {!risk_free_rate} is a valid experiment but produces the distorted
          figure described above, which the README quantifies. *)
}

(** [create ...] validates and constructs a config.

    The checks encode relationships that must hold for the strategy to be
    coherent, not merely for the arithmetic to be defined. *)
let create ~hedge_window ~zscore_window ~entry_threshold ~exit_threshold
    ~stop_loss_threshold ~max_holding_bars ~commission_bps ~slippage_bps
    ~initial_capital ~capital_per_trade_frac ~bars_per_year ~risk_free_rate
    ?(accrue_cash_interest = true) () : (t, error) result =
  let err msg = Error (Config_error msg) in
  if hedge_window < 3 then err "hedge_window must be >= 3"
  else if zscore_window < 2 then err "zscore_window must be >= 2"
  else if not (Float.is_finite entry_threshold) || entry_threshold <= 0. then
    err "entry_threshold must be positive and finite"
  else if not (Float.is_finite exit_threshold) || exit_threshold < 0. then
    err "exit_threshold must be non-negative and finite"
  else if exit_threshold >= entry_threshold then
    (* Without this, the exit band contains the entry band: the engine would
       enter and exit on the same bar forever, paying costs each time. *)
    err
      (Printf.sprintf
         "exit_threshold (%.3f) must be strictly less than entry_threshold \
          (%.3f), otherwise every entry closes immediately"
         exit_threshold entry_threshold)
  else if stop_loss_threshold <= entry_threshold then
    (* A stop inside the entry band stops out every position on the bar it is
       opened. *)
    err
      (Printf.sprintf
         "stop_loss_threshold (%.3f) must be strictly greater than \
          entry_threshold (%.3f)"
         stop_loss_threshold entry_threshold)
  else if max_holding_bars < 1 then err "max_holding_bars must be >= 1"
  else if commission_bps < 0. || not (Float.is_finite commission_bps) then
    err "commission_bps must be non-negative and finite"
  else if slippage_bps < 0. || not (Float.is_finite slippage_bps) then
    err "slippage_bps must be non-negative and finite"
  else if initial_capital <= 0. || not (Float.is_finite initial_capital) then
    err "initial_capital must be positive and finite"
  else if capital_per_trade_frac <= 0. || capital_per_trade_frac > 1. then
    err "capital_per_trade_frac must be in (0, 1]"
  else if bars_per_year <= 0. || not (Float.is_finite bars_per_year) then
    err "bars_per_year must be positive and finite"
  else if not (Float.is_finite risk_free_rate) then
    err "risk_free_rate must be finite"
  else
    Ok
      {
        hedge_window;
        zscore_window;
        entry_threshold;
        exit_threshold;
        stop_loss_threshold;
        max_holding_bars;
        commission_bps;
        slippage_bps;
        initial_capital;
        capital_per_trade_frac;
        bars_per_year;
        risk_free_rate;
        accrue_cash_interest;
      }

(** The parameter set used for the headline backtest in the README.

    These are conventional textbook values (Gatev et al. use a 2-sigma entry;
    0.5-sigma exit and a 3-sigma stop are standard practice), chosen {e before}
    looking at results rather than tuned to make the equity curve attractive.
    Tuning them on the same data that reports the Sharpe would be in-sample
    overfitting, which is the second-most-common way a backtest lies. The
    sensitivity sweep in the README shows performance across a grid so a reader
    can judge how much the headline number depends on this particular choice.

    Costs: 1bp commission + 2bp slippage per leg per side = 6bp per leg
    round-trip, 12bp per round-trip trade across both legs. This is realistic
    for liquid US equities via a retail broker and deliberately not optimistic. *)
let default : t =
  match
    create ~hedge_window:60 ~zscore_window:60 ~entry_threshold:2.0
      ~exit_threshold:0.5 ~stop_loss_threshold:3.5 ~max_holding_bars:60
      ~commission_bps:1.0 ~slippage_bps:2.0 ~initial_capital:100_000.
      ~capital_per_trade_frac:0.25 ~bars_per_year:252. ~risk_free_rate:0.04 ()
  with
  | Ok c -> c
  | Error e ->
      (* Unreachable: the literals above satisfy every check. If this fires it
         is a programmer error in the defaults, not a runtime data condition,
         so failing loudly at startup is correct. *)
      failwith ("Config.default is invalid: " ^ string_of_error e)

(** [warmup_bars c] is the number of leading bars during which no signal can be
    produced, because at least one trailing window is not yet full.

    The engine emits no position before this index. Metrics are still computed
    over the full NAV series (which is flat during warm-up), because excluding
    the warm-up would overstate annualized return by shortening the denominator
    while keeping all the profit. *)
let warmup_bars (c : t) : int = max c.hedge_window c.zscore_window

let to_string (c : t) : string =
  Printf.sprintf
    "hedge_window=%d zscore_window=%d entry=%.2f exit=%.2f stop=%.2f \
     max_hold=%d commission_bps=%.2f slippage_bps=%.2f capital=%.2f \
     frac=%.3f bars_per_year=%.1f rf=%.4f cash_interest=%s"
    c.hedge_window c.zscore_window c.entry_threshold c.exit_threshold
    c.stop_loss_threshold c.max_holding_bars c.commission_bps c.slippage_bps
    c.initial_capital c.capital_per_trade_frac c.bars_per_year c.risk_free_rate
    (if c.accrue_cash_interest then "on" else "off")

(** [rf_per_bar c] is the risk-free rate de-annualized to one bar,
    geometrically: [(1 + rf)^(1/bars_per_year) - 1].

    Geometric rather than the linear [rf / bars_per_year] shortcut. At 4% over
    252 bars the two differ by about 0.3% of the daily rate — immaterial here,
    but the geometric form is unambiguously the correct inverse of compounding
    and costs nothing. Defined once so the cash accrual in {!Backtest} and the
    Sharpe hurdle in {!Metrics} cannot drift apart. *)
let rf_per_bar (c : t) : float =
  Float.pow (1. +. c.risk_free_rate) (1. /. c.bars_per_year) -. 1.
