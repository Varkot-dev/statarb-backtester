(** Portfolio state and the accounting invariant.

    NAV is defined once, here, as

    {[ nav = cash + position_value ]}

    and every reported metric derives from the NAV series. Nothing accumulates
    PnL separately. That matters because the classic backtester bug is a
    "total PnL" counter that drifts away from the actual book — it is updated
    on trade close but not on mark-to-market, or double-counts a cost — and
    since nothing cross-checks it, the error is invisible.

    Here there is only one number, and {!check_reconciliation} asserts every
    bar that the independent path (previous NAV, plus mark-to-market change,
    minus costs) agrees with it. If any module mis-signs a leg or charges a
    cost twice, that test fails. *)

open Types

type t = {
  cash : float;
  position : position;
  realized_pnl : float;  (** Cumulative net PnL of closed trades; reporting only. *)
  total_costs : float;  (** Cumulative transaction costs; reporting only. *)
  trades : trade list;  (** Most recent first. *)
}

let initial (cfg : Config.t) : t =
  {
    cash = cfg.initial_capital;
    position = Flat;
    realized_pnl = 0.;
    total_costs = 0.;
    trades = [];
  }

(** [position_value t ~price_a ~price_b] is the mark-to-market value of the open
    legs, or 0 when flat. *)
let position_value (t : t) ~(price_a : float) ~(price_b : float) : float =
  match t.position with
  | Flat -> 0.
  | Open p -> Execution.position_value p ~price_a ~price_b

(** [nav t ~price_a ~price_b] is the single source of truth for portfolio
    value. *)
let nav (t : t) ~(price_a : float) ~(price_b : float) : float =
  t.cash +. position_value t ~price_a ~price_b

(** [gross_exposure t ~price_a ~price_b] is total absolute notional across
    legs, reported so that the leverage implied by the sizing rule is visible. *)
let gross_exposure (t : t) ~(price_a : float) ~(price_b : float) : float =
  match t.position with
  | Flat -> 0.
  | Open p -> Execution.gross_notional p ~price_a ~price_b

(** [apply_fill t fill] folds an execution result into the portfolio. *)
let apply_fill (t : t) (fill : Execution.fill) : t =
  {
    cash = t.cash +. fill.cash_delta;
    position = fill.new_position;
    realized_pnl =
      (match fill.closed_trade with
      | Some tr -> t.realized_pnl +. tr.pnl_net
      | None -> t.realized_pnl);
    total_costs = t.total_costs +. fill.cost;
    trades =
      (match fill.closed_trade with Some tr -> tr :: t.trades | None -> t.trades);
  }

(** Tolerance for the per-bar reconciliation check, in dollars.

    The two paths being compared are algebraically identical, so any difference
    is floating-point rounding across ~1e5-magnitude values. 1e-6 dollars is
    far above accumulated rounding at this scale and far below any real
    accounting error (the smallest meaningful error — one leg's cost
    mis-charged — is on the order of dollars). *)
let reconciliation_tolerance = 1e-6

(** [check_reconciliation ~prev_nav ~prev_position ~price_a ~price_b ~new_nav
      ~costs_this_bar] verifies the accounting identity for one bar.

    The identity: NAV changes only through (a) mark-to-market on positions held
    {e into} this bar, and (b) costs paid this bar. Opening or closing a
    position at the current bar's prices is NAV-neutral before costs, because
    it exchanges cash for position value at the same price.

    [prev_position] is the position held coming {e into} the bar — the one that
    was exposed to this bar's price move. Using the post-trade position here is
    the subtle version of the bug this check exists to catch. *)
let check_reconciliation ~(prev_nav : float) ~(prev_position : position)
    ~(prev_price_a : float) ~(prev_price_b : float) ~(price_a : float)
    ~(price_b : float) ~(new_nav : float) ~(costs_this_bar : float) :
    (unit, string) result =
  let mtm =
    match prev_position with
    | Flat -> 0.
    | Open p ->
        Execution.position_value p ~price_a ~price_b
        -. Execution.position_value p ~price_a:prev_price_a ~price_b:prev_price_b
  in
  let expected = prev_nav +. mtm -. costs_this_bar in
  let diff = Float.abs (expected -. new_nav) in
  if diff <= reconciliation_tolerance then Ok ()
  else
    Error
      (Printf.sprintf
         "NAV reconciliation failed: expected %.10f (prev %.10f + mtm %.10f - \
          costs %.10f), got %.10f, diff %.3e"
         expected prev_nav mtm costs_this_bar new_nav diff)
