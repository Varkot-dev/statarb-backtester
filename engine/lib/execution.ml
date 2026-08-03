(** Execution model: position sizing, fills, and transaction costs.

    {1 Next-bar execution}

    The single most important property of this module is that a signal computed
    from data through bar [t] fills at the price of bar [t+1]. Same-bar
    execution — deciding to trade because today closed at a 2-sigma spread, and
    then filling at today's close — is lookahead. You could not have known
    today's close before it printed, so you could not have submitted the order.

    This module does not enforce the timing by itself; {!Backtest} does, by
    calling {!apply} with the {e next} bar's prices. What this module does is
    make the convention explicit in its signature: {!apply} takes a
    [fill_price_a]/[fill_price_b] pair distinct from the bar the signal was
    computed on, so a caller cannot pass "the current bar" without noticing.

    {1 Cost model}

    Per leg, per side:

    {[ cost = notional * (commission_bps + slippage_bps) / 10_000 ]}

    Both are charged on entry and again on exit, on both legs. With the default
    1bp commission + 2bp slippage that is 3bp x 2 legs x 2 sides = 12bp of
    notional per round trip. Slippage is modelled as a cost rather than as an
    adjusted fill price; the two are equivalent for PnL and the cost form keeps
    the fill price equal to the observed price, which makes the audit trail
    easier to check against the input data.

    This is a deliberately simple model. It does not include market impact
    (cost rising with size), borrow cost on the short leg, or the bid-ask
    spread widening in stress. Those omissions are listed in the README. *)

open Types

(** Basis points per unit. *)
let bps_denominator = 10_000.

(** [cost_of_notional cfg notional] is the one-leg, one-side transaction cost
    on [notional] dollars. [notional] is taken as an absolute value, so the
    caller need not worry about the sign of a short leg. *)
let cost_of_notional (cfg : Config.t) (notional : float) : float =
  Float.abs notional *. (cfg.commission_bps +. cfg.slippage_bps)
  /. bps_denominator

(** Result of executing an intent on one bar. *)
type fill = {
  new_position : position;
  cash_delta : float;
      (** Change in cash from the trade, {e including} costs. Buying consumes
          cash (negative); selling short releases it (positive). *)
  cost : float;  (** Total transaction cost charged this bar. *)
  closed_trade : trade option;
      (** Present when a position was closed, for the trade log. *)
  event : string;  (** Audit-trail label. *)
}

let no_fill (p : position) : fill =
  { new_position = p; cash_delta = 0.; cost = 0.; closed_trade = None; event = "" }

(** {1 Position sizing}

    Rule: commit [capital_per_trade_frac] of {e initial} capital as the notional
    of leg A, then size leg B by the hedge ratio so the two legs are
    beta-neutral in log-price terms.

    {[
      notional_A = initial_capital * capital_per_trade_frac
      qty_A      = notional_A / price_A
      qty_B      = |beta| * qty_A * (price_A / price_B)
    ]}

    The [price_A / price_B] factor converts the beta — which relates {e log}
    prices, i.e. percentage moves — into a share ratio. Omitting it is a common
    bug: it leaves the dollar exposures unbalanced by the ratio of the two
    price levels, so a "market-neutral" pair of a $500 stock and a $30 stock is
    in fact heavily directional.

    {2 Why initial and not current capital}

    Sizing off current NAV compounds: a strategy that gets lucky early trades
    larger later, which inflates the terminal return without any improvement in
    the underlying edge, and makes the Sharpe sensitive to the {e order} of
    returns. Fixed dollar sizing isolates the per-trade edge, which is what the
    backtest is trying to measure. The consequence — that reported returns do
    not compound — is stated in the README.

    {2 Leverage}

    A pairs trade holds notional in both legs, so committing 25% of capital to
    leg A means roughly 50% gross exposure. Gross exposure is reported in the
    metrics so this is visible rather than hidden. *)
let size_position (cfg : Config.t) ~(price_a : float) ~(price_b : float)
    ~(beta : float) : (Qty.t * Qty.t, error) result =
  let open R in
  if price_a <= 0. then Error (Invalid_price price_a)
  else if price_b <= 0. then Error (Invalid_price price_b)
  else if not (Float.is_finite beta) then
    Error (Degenerate_regression "beta is not finite")
  else
    let notional_a = cfg.initial_capital *. cfg.capital_per_trade_frac in
    let* qty_a = Qty.of_float (notional_a /. price_a) in
    (* |beta|: direction is carried by the position, never by the sign of a
       quantity. A negative beta means the two assets move inversely, in which
       case a "long spread" is long both legs economically; that is a valid
       cointegrating relationship and the sign is handled by the position
       convention, not by a negative share count. *)
    let raw_qty_b =
      Float.abs beta *. Qty.to_float qty_a *. (price_a /. price_b)
    in
    let* qty_b = Qty.of_float raw_qty_b in
    Ok (qty_a, qty_b)

(** [gross_notional p ~price_a ~price_b] is the absolute dollar exposure across
    both legs. *)
let gross_notional (p : open_position) ~(price_a : float) ~(price_b : float) :
    float =
  (Qty.to_float p.qty_a *. price_a) +. (Qty.to_float p.qty_b *. price_b)

(** [position_value p ~price_a ~price_b] is the signed mark-to-market value of
    the open legs.

    Combined with cash this gives NAV. For a long leg the value is positive
    (an asset); for a short leg it is negative (a liability, since closing it
    requires buying back). {!Types.signed_qty} is the only source of the sign. *)
let position_value (p : open_position) ~(price_a : float) ~(price_b : float) :
    float =
  (signed_qty p Leg_a *. price_a) +. (signed_qty p Leg_b *. price_b)

(** [open_cash_delta] is the cash effect of establishing [p] at the given
    prices, before costs: the negative of the position's value. Buying an asset
    converts cash into position value; shorting converts negative position
    value into cash. NAV is therefore unchanged by the act of opening, except
    for the costs — which is exactly the invariant the reconciliation test
    checks. *)
let open_cash_delta (p : open_position) ~(price_a : float) ~(price_b : float) :
    float =
  -.position_value p ~price_a ~price_b

(** [entry_cost cfg p ~price_a ~price_b] is the total cost of establishing [p]:
    one side, both legs. *)
let entry_cost (cfg : Config.t) (p : open_position) ~(price_a : float)
    ~(price_b : float) : float =
  cost_of_notional cfg (Qty.to_float p.qty_a *. price_a)
  +. cost_of_notional cfg (Qty.to_float p.qty_b *. price_b)

(** [apply cfg ~intent ~position ~fill_price_a ~fill_price_b ~fill_index
      ~fill_date ~beta ~z] executes [intent] at the supplied fill prices.

    The caller is responsible for passing the {e next} bar's prices relative to
    the bar the intent was computed on. [beta] and [z] are the values from the
    signal bar, carried through for the trade record. *)
let apply (cfg : Config.t) ~(intent : Signal.intent) ~(position : position)
    ~(fill_price_a : float) ~(fill_price_b : float) ~(fill_index : int)
    ~(fill_date : string) ~(beta : float) ~(z : float)
    ~(entry_dates : (int, string) Hashtbl.t) : (fill, error) result =
  let open R in
  let open_new (dir : direction) : (open_position * float * float, error) result
      =
    let* qty_a, qty_b =
      size_position cfg ~price_a:fill_price_a ~price_b:fill_price_b ~beta
    in
    let* pa = Price.of_float fill_price_a in
    let* pb = Price.of_float fill_price_b in
    let p =
      {
        dir;
        qty_a;
        qty_b;
        entry_price_a = pa;
        entry_price_b = pb;
        entry_index = fill_index;
        entry_hedge_ratio = beta;
        entry_z = z;
      }
    in
    Hashtbl.replace entry_dates fill_index fill_date;
    let cash = open_cash_delta p ~price_a:fill_price_a ~price_b:fill_price_b in
    let cost = entry_cost cfg p ~price_a:fill_price_a ~price_b:fill_price_b in
    Ok (p, cash, cost)
  in
  let close_existing (p : open_position) (reason : exit_reason) :
      (float * float * trade, error) result =
    (* Liquidating returns the mark-to-market value to cash. *)
    let cash = position_value p ~price_a:fill_price_a ~price_b:fill_price_b in
    let cost =
      cost_of_notional cfg (Qty.to_float p.qty_a *. fill_price_a)
      +. cost_of_notional cfg (Qty.to_float p.qty_b *. fill_price_b)
    in
    (* Gross PnL is the change in the value of the signed legs between entry
       and exit fills. Entry costs were already charged at entry, so only the
       exit costs are added here; [costs] on the trade record sums both so the
       trade log is self-contained. *)
    let entry_val =
      position_value p
        ~price_a:(Price.to_float p.entry_price_a)
        ~price_b:(Price.to_float p.entry_price_b)
    in
    let exit_val = position_value p ~price_a:fill_price_a ~price_b:fill_price_b in
    let pnl_gross = exit_val -. entry_val in
    let ecost =
      entry_cost cfg p
        ~price_a:(Price.to_float p.entry_price_a)
        ~price_b:(Price.to_float p.entry_price_b)
    in
    let total_costs = ecost +. cost in
    let entry_date =
      match Hashtbl.find_opt entry_dates p.entry_index with
      | Some d -> d
      | None -> ""
    in
    let t =
      {
        t_dir = p.dir;
        entry_index = p.entry_index;
        exit_index = fill_index;
        entry_date;
        exit_date = fill_date;
        t_entry_z = p.entry_z;
        t_exit_z = z;
        pnl_gross;
        costs = total_costs;
        pnl_net = pnl_gross -. total_costs;
        reason;
        holding_bars = fill_index - p.entry_index;
      }
    in
    Ok (cash, cost, t)
  in
  match (intent, position) with
  | Signal.Hold, _ -> Ok (no_fill position)
  | Signal.Enter dir, Flat ->
      let* p, cash, cost = open_new dir in
      Ok
        {
          new_position = Open p;
          cash_delta = cash -. cost;
          cost;
          closed_trade = None;
          event = "entry:" ^ string_of_direction dir;
        }
  | Signal.Enter _, Open _ ->
      (* Defensive: [Signal.decide] only emits [Enter] when flat. Reaching here
         means the signal layer and the execution layer disagree about state,
         which is a bug rather than a data condition. Holding is the safe
         response and the event label makes it visible in the audit trail. *)
      Ok { (no_fill position) with event = "ignored_enter_while_open" }
  | Signal.Exit reason, Open p ->
      let* cash, cost, t = close_existing p reason in
      Ok
        {
          new_position = Flat;
          cash_delta = cash -. cost;
          cost;
          closed_trade = Some t;
          event = "exit:" ^ string_of_exit_reason reason;
        }
  | Signal.Exit _, Flat -> Ok (no_fill position)
  | Signal.Flip dir, Open p ->
      let* cash_close, cost_close, t = close_existing p Reversion in
      let* np, cash_open, cost_open = open_new dir in
      Ok
        {
          new_position = Open np;
          cash_delta = cash_close +. cash_open -. cost_close -. cost_open;
          cost = cost_close +. cost_open;
          closed_trade = Some t;
          event = "flip:" ^ string_of_direction dir;
        }
  | Signal.Flip dir, Flat ->
      let* p, cash, cost = open_new dir in
      Ok
        {
          new_position = Open p;
          cash_delta = cash -. cost;
          cost;
          closed_trade = None;
          event = "entry:" ^ string_of_direction dir;
        }
