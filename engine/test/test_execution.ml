(** Execution, sizing, cost, and PnL accounting invariants.

    The centrepiece here is {!test_nav_reconciles_every_bar}: cash plus
    position value must equal NAV at every bar, and NAV must change only
    through mark-to-market and costs. A backtester that fails this is
    reporting a number that does not correspond to any portfolio. *)

open Statarb
open Statarb.Types

let eps = 1e-8

(** {1 Sizing} *)

(** With beta = 1 and equal prices, the two legs have equal share counts.

    notional_A = 100000 * 0.25 = 25000; qty_A = 25000/50 = 500.
    qty_B = |1| * 500 * (50/50) = 500. *)
let test_sizing_equal_prices_unit_beta () =
  let cfg = Fixtures.test_config () in
  let qa, qb =
    Fixtures.get_ok
      (Execution.size_position cfg ~price_a:50. ~price_b:50. ~beta:1.0)
  in
  Alcotest.(check (float eps)) "qty_a" 500. (Qty.to_float qa);
  Alcotest.(check (float eps)) "qty_b" 500. (Qty.to_float qb)

(** {b The price-ratio correction.}

    With A at $500 and B at $30 and beta = 1, a naive [qty_B = beta * qty_A]
    would give equal share counts and wildly unequal dollar exposures. The
    correct sizing equalises {e notional}:

    notional_A = 25000; qty_A = 25000/500 = 50.
    qty_B = 1 * 50 * (500/30) = 833.33...
    notional_B = 833.33 * 30 = 25000. Equal, as required. *)
let test_sizing_applies_the_price_ratio () =
  let cfg = Fixtures.test_config () in
  let qa, qb =
    Fixtures.get_ok
      (Execution.size_position cfg ~price_a:500. ~price_b:30. ~beta:1.0)
  in
  let notional_a = Qty.to_float qa *. 500. in
  let notional_b = Qty.to_float qb *. 30. in
  Alcotest.(check (float 1e-6)) "notionals are equal at beta=1" notional_a
    notional_b;
  Alcotest.(check (float 1e-6)) "notional_a" 25000. notional_a

(** At beta = 2, leg B carries twice the notional of leg A. *)
let test_sizing_scales_with_beta () =
  let cfg = Fixtures.test_config () in
  let qa, qb =
    Fixtures.get_ok
      (Execution.size_position cfg ~price_a:100. ~price_b:100. ~beta:2.0)
  in
  Alcotest.(check (float 1e-6)) "leg B notional is 2x leg A"
    (2. *. Qty.to_float qa *. 100.)
    (Qty.to_float qb *. 100.)

(** A negative beta is a valid cointegrating relationship; the magnitude is
    used for sizing and the sign is carried by the position, so sizing must
    match the positive case. *)
let test_sizing_uses_absolute_beta () =
  let cfg = Fixtures.test_config () in
  let _, qb_pos =
    Fixtures.get_ok
      (Execution.size_position cfg ~price_a:100. ~price_b:50. ~beta:1.5)
  in
  let _, qb_neg =
    Fixtures.get_ok
      (Execution.size_position cfg ~price_a:100. ~price_b:50. ~beta:(-1.5))
  in
  Alcotest.(check (float eps)) "sizing uses |beta|" (Qty.to_float qb_pos)
    (Qty.to_float qb_neg)

let test_sizing_rejects_bad_prices () =
  let cfg = Fixtures.test_config () in
  (match Execution.size_position cfg ~price_a:0. ~price_b:50. ~beta:1. with
  | Error (Invalid_price _) -> ()
  | _ -> Alcotest.fail "zero price_a should be rejected");
  match Execution.size_position cfg ~price_a:50. ~price_b:(-1.) ~beta:1. with
  | Error (Invalid_price _) -> ()
  | _ -> Alcotest.fail "negative price_b should be rejected"

(** {1 Costs} *)

(** 1bp commission + 2bp slippage = 3bp of notional, per leg per side.
    On $25,000: 25000 * 3/10000 = $7.50. *)
let test_cost_hand_computed () =
  let cfg = Fixtures.test_config ~commission_bps:1.0 ~slippage_bps:2.0 () in
  Alcotest.(check (float eps)) "3bp on 25000" 7.50
    (Execution.cost_of_notional cfg 25000.)

(** Cost is on absolute notional, so a short leg costs the same as a long one. *)
let test_cost_uses_absolute_notional () =
  let cfg = Fixtures.test_config () in
  Alcotest.(check (float eps)) "cost of a short leg"
    (Execution.cost_of_notional cfg 25000.)
    (Execution.cost_of_notional cfg (-25000.))

let test_zero_cost_config_charges_nothing () =
  let cfg = Fixtures.zero_cost_config () in
  Alcotest.(check (float eps)) "zero cost" 0.
    (Execution.cost_of_notional cfg 1e6)

(** {1 Position sign conventions} *)

let mk_position ~dir ~qa ~qb ~pa ~pb : open_position =
  {
    dir;
    qty_a = Fixtures.get_ok (Qty.of_float qa);
    qty_b = Fixtures.get_ok (Qty.of_float qb);
    entry_price_a = Fixtures.get_ok (Price.of_float pa);
    entry_price_b = Fixtures.get_ok (Price.of_float pb);
    entry_index = 0;
    entry_hedge_ratio = 1.0;
    entry_z = -2.5;
  }

(** Long spread = long A, short B. *)
let test_signed_qty_long_spread () =
  let p = mk_position ~dir:Long_spread ~qa:100. ~qb:200. ~pa:50. ~pb:25. in
  Alcotest.(check (float eps)) "leg A is long" 100. (signed_qty p Leg_a);
  Alcotest.(check (float eps)) "leg B is short" (-200.) (signed_qty p Leg_b)

(** Short spread = short A, long B. *)
let test_signed_qty_short_spread () =
  let p = mk_position ~dir:Short_spread ~qa:100. ~qb:200. ~pa:50. ~pb:25. in
  Alcotest.(check (float eps)) "leg A is short" (-100.) (signed_qty p Leg_a);
  Alcotest.(check (float eps)) "leg B is long" 200. (signed_qty p Leg_b)

(** Position value: long A (100 @ 50 = +5000) short B (200 @ 25 = -5000) nets
    to zero at entry prices — the position is dollar-neutral by construction. *)
let test_position_value_is_dollar_neutral_at_entry () =
  let p = mk_position ~dir:Long_spread ~qa:100. ~qb:200. ~pa:50. ~pb:25. in
  Alcotest.(check (float eps)) "dollar-neutral at entry" 0.
    (Execution.position_value p ~price_a:50. ~price_b:25.)

(** If A rises 10% and B is unchanged, a long spread gains 10% of leg A's
    notional: 100 shares * $5 = $500. *)
let test_position_value_moves_with_prices () =
  let p = mk_position ~dir:Long_spread ~qa:100. ~qb:200. ~pa:50. ~pb:25. in
  Alcotest.(check (float eps)) "long spread gains when A rises" 500.
    (Execution.position_value p ~price_a:55. ~price_b:25.);
  Alcotest.(check (float eps)) "long spread loses when B rises" (-500.)
    (Execution.position_value p ~price_a:50. ~price_b:27.5)

(** Gross notional sums both legs' absolute exposure: 5000 + 5000 = 10000.
    With 25% of $100k committed to leg A, gross exposure is ~50% of capital —
    the leverage implied by the sizing rule, made visible. *)
let test_gross_notional () =
  let p = mk_position ~dir:Long_spread ~qa:100. ~qb:200. ~pa:50. ~pb:25. in
  Alcotest.(check (float eps)) "gross notional" 10000.
    (Execution.gross_notional p ~price_a:50. ~price_b:25.)

(** {1 The accounting invariant} *)

(** {b The PnL reconciliation test.}

    Runs a full backtest and independently re-derives NAV at every bar from
    cash and position value, then checks NAV changes only via mark-to-market
    and costs. The engine also asserts this internally and aborts on failure;
    this test verifies from the outside, against the emitted audit trail,
    which is what a third party would check. *)
let test_nav_reconciles_every_bar () =
  let cfg = Fixtures.test_config () in
  let series =
    Fixtures.cointegrated ~n:600 ~seed:99 ~beta:1.15 ~half_life:12.
      ~sigma_spread:0.03 ~sigma_common:0.015
  in
  let res = Fixtures.get_ok (Backtest.run cfg series) in
  let bars = Array.of_list res.bars in
  Alcotest.(check bool) "backtest produced bars" true (Array.length bars > 0);
  (* Identity 1: NAV = cash + position value, at every bar. *)
  Array.iteri
    (fun i r ->
      Alcotest.(check (float 1e-6))
        (Printf.sprintf "NAV = cash + position value at bar %d" i)
        (r.r_cash +. r.r_position_value) r.r_nav)
    bars;
  (* Identity 2: NAV changes only through mark-to-market on the position held
     into the bar, interest accrued on cash, and costs paid in the bar.
     Reconstructed here from the previous bar's signed quantities and the price
     change. *)
  for i = 1 to Array.length bars - 1 do
    let prev = bars.(i - 1) and cur = bars.(i) in
    let mtm =
      (prev.r_qty_a *. (cur.r_price_a -. prev.r_price_a))
      +. (prev.r_qty_b *. (cur.r_price_b -. prev.r_price_b))
    in
    let expected =
      prev.r_nav +. mtm +. cur.r_interest_this_bar -. cur.r_costs_this_bar
    in
    Alcotest.(check (float 1e-6))
      (Printf.sprintf "NAV evolution at bar %d (%s)" i cur.r_date)
      expected cur.r_nav
  done

(** {b Conservation of money.} Every dollar of NAV change is accounted for by
    either a trade or interest — nothing appears or vanishes in between.

    Run with interest disabled so the identity reduces to the cleanest possible
    statement (NAV change = sum of trade PnL); the companion test below then
    checks the identity with interest on. Splitting them means a failure points
    at which term is wrong rather than at the sum. *)
let test_zero_cost_nav_equals_sum_of_trade_pnl () =
  let cfg =
    Fixtures.test_config ~commission_bps:0. ~slippage_bps:0.
      ~accrue_cash_interest:false ()
  in
  let series =
    Fixtures.cointegrated ~n:600 ~seed:55 ~beta:1.0 ~half_life:10.
      ~sigma_spread:0.03 ~sigma_common:0.01
  in
  let res = Fixtures.get_ok (Backtest.run cfg series) in
  let nav_change = res.metrics.final_nav -. res.metrics.initial_nav in
  let sum_pnl =
    List.fold_left (fun a (t : trade) -> a +. t.pnl_net) 0. res.trades
  in
  Alcotest.(check bool) "some trades occurred" true (List.length res.trades > 0);
  Alcotest.(check (float 0.)) "no interest was accrued" 0.
    res.metrics.total_interest;
  Alcotest.(check (float 1e-6))
    "NAV change equals the sum of net trade PnL at zero cost and zero interest"
    nav_change sum_pnl

(** The same conservation law with interest on: NAV change is trade PnL plus
    interest, exactly. *)
let test_nav_change_equals_trade_pnl_plus_interest () =
  let cfg =
    Fixtures.test_config ~commission_bps:0. ~slippage_bps:0.
      ~risk_free_rate:0.04 ()
  in
  let series =
    Fixtures.cointegrated ~n:600 ~seed:55 ~beta:1.0 ~half_life:10.
      ~sigma_spread:0.03 ~sigma_common:0.01
  in
  let res = Fixtures.get_ok (Backtest.run cfg series) in
  let nav_change = res.metrics.final_nav -. res.metrics.initial_nav in
  let sum_pnl =
    List.fold_left (fun a (t : trade) -> a +. t.pnl_net) 0. res.trades
  in
  Alcotest.(check bool) "interest was accrued" true
    (res.metrics.total_interest > 0.);
  Alcotest.(check (float 1e-6))
    "NAV change equals trade PnL plus interest" nav_change
    (sum_pnl +. res.metrics.total_interest)

(** Costs strictly reduce net PnL relative to gross. *)
let test_costs_reduce_pnl () =
  let series =
    Fixtures.cointegrated ~n:600 ~seed:77 ~beta:1.0 ~half_life:10.
      ~sigma_spread:0.03 ~sigma_common:0.01
  in
  let free = Fixtures.get_ok (Backtest.run (Fixtures.zero_cost_config ()) series) in
  let costly =
    Fixtures.get_ok
      (Backtest.run
         (Fixtures.test_config ~commission_bps:5. ~slippage_bps:5. ())
         series)
  in
  Alcotest.(check bool) "zero-cost run paid nothing" true
    (free.metrics.total_costs = 0.);
  Alcotest.(check bool) "costly run paid something" true
    (costly.metrics.total_costs > 0.);
  Alcotest.(check bool)
    (Printf.sprintf "costs reduce final NAV (free %.2f vs costly %.2f)"
       free.metrics.final_nav costly.metrics.final_nav)
    true
    (costly.metrics.final_nav < free.metrics.final_nav)

(** Every trade's net PnL is gross minus costs, exactly. A drift here would
    mean costs are counted somewhere else too. *)
let test_trade_pnl_decomposition () =
  let cfg = Fixtures.test_config () in
  let series =
    Fixtures.cointegrated ~n:500 ~seed:13 ~beta:1.05 ~half_life:11.
      ~sigma_spread:0.03 ~sigma_common:0.012
  in
  let res = Fixtures.get_ok (Backtest.run cfg series) in
  Alcotest.(check bool) "trades exist" true (List.length res.trades > 0);
  List.iteri
    (fun i (t : trade) ->
      Alcotest.(check (float 1e-9))
        (Printf.sprintf "trade %d: net = gross - costs" i)
        (t.pnl_gross -. t.costs) t.pnl_net;
      Alcotest.(check bool)
        (Printf.sprintf "trade %d has non-negative costs" i)
        true (t.costs >= 0.);
      Alcotest.(check bool)
        (Printf.sprintf "trade %d has a positive holding period" i)
        true (t.holding_bars > 0))
    res.trades

(** The engine must never hold two positions at once, and every bar's position
    label must be one of the three legal values. *)
let test_position_is_always_flat_or_single () =
  let cfg = Fixtures.test_config () in
  let series =
    Fixtures.cointegrated ~n:400 ~seed:8 ~beta:1.0 ~half_life:9.
      ~sigma_spread:0.04 ~sigma_common:0.01
  in
  let res = Fixtures.get_ok (Backtest.run cfg series) in
  List.iter
    (fun r ->
      Alcotest.(check bool)
        (Printf.sprintf "legal position label at bar %d: %s" r.r_index
           r.r_position)
        true
        (List.mem r.r_position [ "flat"; "long_spread"; "short_spread" ]);
      (* Signs must be opposite (or both zero): a pairs position is always one
         leg long and one leg short. *)
      if r.r_position <> "flat" then
        Alcotest.(check bool)
          (Printf.sprintf "legs have opposite signs at bar %d" r.r_index)
          true
          (r.r_qty_a *. r.r_qty_b < 0.)
      else
        Alcotest.(check bool)
          (Printf.sprintf "flat means zero quantities at bar %d" r.r_index)
          true
          (r.r_qty_a = 0. && r.r_qty_b = 0.))
    res.bars

(** {1 Cash interest accrual}

    Idle cash earns the risk-free rate. Without this, a strategy that is flat
    most of the time is charged a risk-free hurdle on its whole NAV while
    earning nothing on the cash it holds, and the resulting Sharpe measures how
    often it was flat rather than whether it had an edge. *)

(** Interest is credited on cash only, at the geometrically de-annualized rate.

    $100,000 at 4% annual over one bar of 252:
    rf_bar = 1.04^(1/252) - 1 = 1.55654e-4, so interest = $15.5654. *)
let test_interest_is_credited_on_cash () =
  let cfg = Fixtures.test_config ~risk_free_rate:0.04 () in
  let p = Portfolio.initial cfg in
  let p', interest = Portfolio.accrue_interest p cfg in
  let expected = 100_000. *. (Float.pow 1.04 (1. /. 252.) -. 1.) in
  Alcotest.(check (float 1e-9)) "one bar of interest" expected interest;
  Alcotest.(check (float 1e-9)) "cash increased by the interest"
    (100_000. +. expected) p'.cash

(** De-annualization is geometric, not linear: compounding [rf_bar] over
    [bars_per_year] must return exactly the annual rate. *)
let test_interest_compounds_to_the_annual_rate () =
  let cfg = Fixtures.test_config ~risk_free_rate:0.04 () in
  let rf_bar = Config.rf_per_bar cfg in
  Alcotest.(check (float 1e-12))
    "compounding 252 bars recovers the annual rate" 1.04
    (Float.pow (1. +. rf_bar) 252.);
  (* And it differs from the linear shortcut, so a silent switch would fail. *)
  Alcotest.(check bool) "geometric differs from linear" true
    (Float.abs (rf_bar -. (0.04 /. 252.)) > 1e-9)

let test_interest_can_be_disabled () =
  let cfg = Fixtures.test_config ~risk_free_rate:0.04 ~accrue_cash_interest:false () in
  let p = Portfolio.initial cfg in
  let p', interest = Portfolio.accrue_interest p cfg in
  Alcotest.(check (float 0.)) "no interest when disabled" 0. interest;
  Alcotest.(check (float 0.)) "cash unchanged" 100_000. p'.cash

let test_zero_rate_accrues_nothing () =
  let cfg = Fixtures.test_config ~risk_free_rate:0. () in
  let _, interest = Portfolio.accrue_interest (Portfolio.initial cfg) cfg in
  Alcotest.(check (float 0.)) "no interest at a zero rate" 0. interest

(** With interest on and no trading, NAV must compound at exactly the
    risk-free rate — the cleanest possible check that the accrual is correct.

    An entry threshold of 100 sigmas is never reached, so the strategy stays
    flat for the whole sample and NAV is pure interest. *)
let test_flat_portfolio_compounds_at_the_risk_free_rate () =
  let cfg =
    Fixtures.test_config ~entry_threshold:100. ~stop_loss_threshold:200.
      ~risk_free_rate:0.04 ()
  in
  let series =
    Fixtures.cointegrated ~n:300 ~seed:71 ~beta:1.0 ~half_life:12.
      ~sigma_spread:0.03 ~sigma_common:0.012
  in
  let res = Fixtures.get_ok (Backtest.run cfg series) in
  Alcotest.(check int) "no trades at an unreachable threshold" 0
    res.metrics.n_trades;
  (* 300 bars, of which bar 0 accrues nothing: 299 compounding periods. *)
  let expected = 100_000. *. Float.pow (1. +. Config.rf_per_bar cfg) 299. in
  Alcotest.(check (float 1e-6))
    "a flat portfolio compounds at exactly the risk-free rate" expected
    res.metrics.final_nav;
  (* And its Sharpe is therefore ~0: it earns exactly the hurdle, no more. *)
  Alcotest.(check bool)
    (Printf.sprintf "Sharpe of a pure-cash portfolio is ~0 (got %.6f)"
       res.metrics.sharpe_ratio)
    true
    (Float.abs res.metrics.sharpe_ratio < 1e-6)

(** Interest appears in the audit trail and sums to the reported total. *)
let test_interest_is_reported_and_reconciles () =
  let cfg = Fixtures.test_config ~risk_free_rate:0.04 () in
  let series =
    Fixtures.cointegrated ~n:400 ~seed:73 ~beta:1.0 ~half_life:12.
      ~sigma_spread:0.03 ~sigma_common:0.012
  in
  let res = Fixtures.get_ok (Backtest.run cfg series) in
  let summed =
    List.fold_left (fun a r -> a +. r.r_interest_this_bar) 0. res.bars
  in
  Alcotest.(check bool) "some interest was earned" true (summed > 0.);
  Alcotest.(check (float 1e-6)) "per-bar interest sums to the reported total"
    summed res.metrics.total_interest

(** {1 Type-level guarantees}

    These do not test runtime behaviour so much as document that the compiler
    already rules the state out. The commented expression below does not
    compile; the runtime checks confirm the constructors behave as claimed. *)

(** [Qty.of_float] refuses non-positive and non-finite magnitudes, so a
    negative share count cannot enter the system. *)
let test_qty_rejects_invalid () =
  List.iter
    (fun (x, label) ->
      match Qty.of_float x with
      | Error (Invalid_quantity _) -> ()
      | _ -> Alcotest.failf "%s should be rejected" label)
    [
      (0., "zero");
      (-1., "negative");
      (Float.nan, "nan");
      (Float.infinity, "infinity");
      (Float.neg_infinity, "negative infinity");
    ];
  Alcotest.(check (float eps)) "a valid quantity round-trips" 42.
    (Qty.to_float (Fixtures.get_ok (Qty.of_float 42.)))

let test_price_rejects_invalid () =
  List.iter
    (fun (x, label) ->
      match Price.of_float x with
      | Error (Invalid_price _) -> ()
      | _ -> Alcotest.failf "%s should be rejected" label)
    [ (0., "zero"); (-1., "negative"); (Float.nan, "nan") ]

(** A position is [Flat] or [Open] of exactly one direction. There is no
    constructor for "both", which is the type-level statement that the illegal
    state is unrepresentable. This test simply confirms the two cases are
    exhaustive at runtime. *)
let test_position_has_exactly_two_states () =
  let flat = Flat in
  let opened = Open (mk_position ~dir:Long_spread ~qa:1. ~qb:1. ~pa:1. ~pb:1.) in
  Alcotest.(check bool) "Flat is flat" true (is_flat flat);
  Alcotest.(check bool) "Open is not flat" false (is_flat opened);
  (* Exhaustive match: adding a third state would fail to compile here. *)
  let label = function Flat -> "flat" | Open p -> string_of_direction p.dir in
  Alcotest.(check string) "flat label" "flat" (label flat);
  Alcotest.(check string) "open label" "long_spread" (label opened)

let test_opposite_is_an_involution () =
  Alcotest.(check bool) "opposite twice is identity" true
    (opposite (opposite Long_spread) = Long_spread
    && opposite (opposite Short_spread) = Short_spread);
  Alcotest.(check bool) "opposite actually flips" true
    (opposite Long_spread = Short_spread)

let tests =
  [
    ("sizing: equal prices, unit beta", `Quick, test_sizing_equal_prices_unit_beta);
    ("sizing applies the price ratio", `Quick, test_sizing_applies_the_price_ratio);
    ("sizing scales with beta", `Quick, test_sizing_scales_with_beta);
    ("sizing uses |beta|", `Quick, test_sizing_uses_absolute_beta);
    ("sizing rejects bad prices", `Quick, test_sizing_rejects_bad_prices);
    ("cost hand-computed", `Quick, test_cost_hand_computed);
    ("cost uses absolute notional", `Quick, test_cost_uses_absolute_notional);
    ("zero-cost config charges nothing", `Quick, test_zero_cost_config_charges_nothing);
    ("signed qty: long spread", `Quick, test_signed_qty_long_spread);
    ("signed qty: short spread", `Quick, test_signed_qty_short_spread);
    ("position is dollar-neutral at entry", `Quick,
     test_position_value_is_dollar_neutral_at_entry);
    ("position value moves with prices", `Quick, test_position_value_moves_with_prices);
    ("gross notional", `Quick, test_gross_notional);
    ("NAV reconciles every bar (PnL invariant)", `Quick, test_nav_reconciles_every_bar);
    ("zero-cost NAV equals sum of trade PnL", `Quick,
     test_zero_cost_nav_equals_sum_of_trade_pnl);
    ("NAV change equals trade PnL plus interest", `Quick,
     test_nav_change_equals_trade_pnl_plus_interest);
    ("costs reduce PnL", `Quick, test_costs_reduce_pnl);
    ("trade PnL decomposition", `Quick, test_trade_pnl_decomposition);
    ("position is always flat or a single direction", `Quick,
     test_position_is_always_flat_or_single);
    ("interest is credited on cash", `Quick, test_interest_is_credited_on_cash);
    ("interest compounds to the annual rate", `Quick,
     test_interest_compounds_to_the_annual_rate);
    ("interest can be disabled", `Quick, test_interest_can_be_disabled);
    ("a zero rate accrues nothing", `Quick, test_zero_rate_accrues_nothing);
    ("a flat portfolio compounds at the risk-free rate", `Quick,
     test_flat_portfolio_compounds_at_the_risk_free_rate);
    ("interest is reported and reconciles", `Quick,
     test_interest_is_reported_and_reconciles);
    ("Qty rejects invalid magnitudes", `Quick, test_qty_rejects_invalid);
    ("Price rejects invalid prices", `Quick, test_price_rejects_invalid);
    ("position has exactly two states", `Quick, test_position_has_exactly_two_states);
    ("opposite is an involution", `Quick, test_opposite_is_an_involution);
  ]
