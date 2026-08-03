(** Signal decision rules and threshold behaviour. *)

open Statarb
open Statarb.Types

let mk_snapshot ?(beta = 1.0) ?(spread = 0.0) (z : float option) : Signal.snapshot =
  { Signal.fit = { Ols.alpha = 0.; beta; r_squared = 0.9 }; spread; z }

let mk_open ?(dir = Long_spread) ?(entry_index = 0) () : position =
  let q = Fixtures.get_ok (Qty.of_float 100.) in
  let p = Fixtures.get_ok (Price.of_float 50.) in
  Open
    {
      dir;
      qty_a = q;
      qty_b = q;
      entry_price_a = p;
      entry_price_b = p;
      entry_index;
      entry_hedge_ratio = 1.0;
      entry_z = 0.;
    }

let cfg = Fixtures.test_config ()

(** A z-score below the negative entry threshold means the spread is unusually
    low; the strategy goes long the spread expecting it to rise. *)
let test_enters_long_on_low_zscore () =
  let i =
    Signal.decide ~snapshot:(mk_snapshot (Some (-2.5))) ~position:Flat
      ~bar_index:100 cfg
  in
  Alcotest.(check string) "enters long spread" "enter:long_spread"
    (Signal.string_of_intent i)

let test_enters_short_on_high_zscore () =
  let i =
    Signal.decide ~snapshot:(mk_snapshot (Some 2.5)) ~position:Flat
      ~bar_index:100 cfg
  in
  Alcotest.(check string) "enters short spread" "enter:short_spread"
    (Signal.string_of_intent i)

(** Inside the band, nothing happens. *)
let test_no_entry_inside_the_band () =
  List.iter
    (fun z ->
      let i =
        Signal.decide ~snapshot:(mk_snapshot (Some z)) ~position:Flat
          ~bar_index:100 cfg
      in
      Alcotest.(check string)
        (Printf.sprintf "no entry at z=%.2f" z)
        "hold" (Signal.string_of_intent i))
    [ 0.; 0.5; 1.0; 1.9; -1.9; -0.5 ]

(** The entry threshold is inclusive: exactly 2.0 triggers. Stating this in a
    test pins the boundary so a later refactor cannot silently change it. *)
let test_entry_threshold_is_inclusive () =
  let i =
    Signal.decide ~snapshot:(mk_snapshot (Some cfg.entry_threshold))
      ~position:Flat ~bar_index:100 cfg
  in
  Alcotest.(check string) "z exactly at the threshold enters" "enter:short_spread"
    (Signal.string_of_intent i)

(** Reversion: |z| back inside the exit band closes the position. *)
let test_exits_on_reversion () =
  let i =
    Signal.decide ~snapshot:(mk_snapshot (Some 0.3))
      ~position:(mk_open ~dir:Long_spread ()) ~bar_index:10 cfg
  in
  Alcotest.(check string) "exits on reversion" "exit:reversion"
    (Signal.string_of_intent i)

(** Stop-loss for a long-spread position fires when z falls {e further}
    negative, not when it rises. Direction-awareness is the point. *)
let test_stop_loss_is_direction_aware () =
  let long_pos = mk_open ~dir:Long_spread () in
  let short_pos = mk_open ~dir:Short_spread () in
  (* Long spread was entered at very negative z; adverse move is more negative. *)
  Alcotest.(check string) "long spread stops on a large negative z"
    "exit:stop_loss"
    (Signal.string_of_intent
       (Signal.decide ~snapshot:(mk_snapshot (Some (-4.0))) ~position:long_pos
          ~bar_index:10 cfg));
  (* A large positive z is favourable for a long spread — that is profit, not a
     stop. It exceeds the exit band, so the position is simply held. *)
  Alcotest.(check string) "long spread does not stop on a large positive z"
    "hold"
    (Signal.string_of_intent
       (Signal.decide ~snapshot:(mk_snapshot (Some 3.0)) ~position:long_pos
          ~bar_index:10 cfg));
  Alcotest.(check string) "short spread stops on a large positive z"
    "exit:stop_loss"
    (Signal.string_of_intent
       (Signal.decide ~snapshot:(mk_snapshot (Some 4.0)) ~position:short_pos
          ~bar_index:10 cfg));
  Alcotest.(check string) "short spread does not stop on a large negative z"
    "hold"
    (Signal.string_of_intent
       (Signal.decide ~snapshot:(mk_snapshot (Some (-3.0))) ~position:short_pos
          ~bar_index:10 cfg))

(** The holding cap is a hard bound. *)
let test_max_holding_forces_exit () =
  let pos = mk_open ~dir:Long_spread ~entry_index:0 () in
  (* One bar short of the cap: still held (z is outside the exit band). *)
  Alcotest.(check string) "held below the cap" "hold"
    (Signal.string_of_intent
       (Signal.decide ~snapshot:(mk_snapshot (Some (-1.5))) ~position:pos
          ~bar_index:(cfg.max_holding_bars - 1) cfg));
  Alcotest.(check string) "exits at the cap" "exit:max_holding"
    (Signal.string_of_intent
       (Signal.decide ~snapshot:(mk_snapshot (Some (-1.5))) ~position:pos
          ~bar_index:cfg.max_holding_bars cfg))

(** Precedence: stop-loss is evaluated before max-holding, so a position that is
    both stopped out and at its holding cap reports the loss reason. Booking the
    more favourable reason would understate the stop-loss count. *)
let test_stop_loss_takes_precedence_over_max_holding () =
  let pos = mk_open ~dir:Long_spread ~entry_index:0 () in
  Alcotest.(check string) "stop-loss wins over max-holding" "exit:stop_loss"
    (Signal.string_of_intent
       (Signal.decide ~snapshot:(mk_snapshot (Some (-5.0))) ~position:pos
          ~bar_index:(cfg.max_holding_bars + 10) cfg))

(** An undefined z-score is not evidence to act. The position is held, not
    liquidated, and no new position is opened. *)
let test_undefined_zscore_holds () =
  Alcotest.(check string) "flat and undefined z stays flat" "hold"
    (Signal.string_of_intent
       (Signal.decide ~snapshot:(mk_snapshot None) ~position:Flat ~bar_index:10
          cfg));
  Alcotest.(check string) "open and undefined z holds" "hold"
    (Signal.string_of_intent
       (Signal.decide ~snapshot:(mk_snapshot None)
          ~position:(mk_open ()) ~bar_index:10 cfg))

(** {1 Config validation}

    Each of these rejects a configuration that is arithmetically fine but
    strategically incoherent. *)

let test_config_rejects_exit_above_entry () =
  match
    Config.create ~hedge_window:20 ~zscore_window:20 ~entry_threshold:1.0
      ~exit_threshold:2.0 ~stop_loss_threshold:3.0 ~max_holding_bars:10
      ~commission_bps:1. ~slippage_bps:1. ~initial_capital:1000.
      ~capital_per_trade_frac:0.5 ~bars_per_year:252. ~risk_free_rate:0. ()
  with
  | Error (Config_error _) -> ()
  | _ -> Alcotest.fail "exit >= entry should be rejected"

let test_config_rejects_stop_below_entry () =
  match
    Config.create ~hedge_window:20 ~zscore_window:20 ~entry_threshold:2.0
      ~exit_threshold:0.5 ~stop_loss_threshold:1.5 ~max_holding_bars:10
      ~commission_bps:1. ~slippage_bps:1. ~initial_capital:1000.
      ~capital_per_trade_frac:0.5 ~bars_per_year:252. ~risk_free_rate:0. ()
  with
  | Error (Config_error _) -> ()
  | _ -> Alcotest.fail "stop <= entry should be rejected"

let test_config_rejects_bad_capital () =
  List.iter
    (fun (cap, frac, label) ->
      match
        Config.create ~hedge_window:20 ~zscore_window:20 ~entry_threshold:2.0
          ~exit_threshold:0.5 ~stop_loss_threshold:3.0 ~max_holding_bars:10
          ~commission_bps:1. ~slippage_bps:1. ~initial_capital:cap
          ~capital_per_trade_frac:frac ~bars_per_year:252. ~risk_free_rate:0. ()
      with
      | Error (Config_error _) -> ()
      | _ -> Alcotest.failf "%s should be rejected" label)
    [
      (0., 0.5, "zero capital");
      (-100., 0.5, "negative capital");
      (1000., 0., "zero fraction");
      (1000., 1.5, "fraction above 1");
    ]

let test_config_rejects_negative_costs () =
  match
    Config.create ~hedge_window:20 ~zscore_window:20 ~entry_threshold:2.0
      ~exit_threshold:0.5 ~stop_loss_threshold:3.0 ~max_holding_bars:10
      ~commission_bps:(-1.) ~slippage_bps:1. ~initial_capital:1000.
      ~capital_per_trade_frac:0.5 ~bars_per_year:252. ~risk_free_rate:0. ()
  with
  | Error (Config_error _) -> ()
  | _ -> Alcotest.fail "negative commission should be rejected"

let test_default_config_is_valid () =
  let c = Config.default in
  Alcotest.(check bool) "entry above exit" true (c.entry_threshold > c.exit_threshold);
  Alcotest.(check bool) "stop above entry" true
    (c.stop_loss_threshold > c.entry_threshold);
  Alcotest.(check int) "warmup is the larger window" 60 (Config.warmup_bars c)

let tests =
  [
    ("enters long on a low z-score", `Quick, test_enters_long_on_low_zscore);
    ("enters short on a high z-score", `Quick, test_enters_short_on_high_zscore);
    ("no entry inside the band", `Quick, test_no_entry_inside_the_band);
    ("entry threshold is inclusive", `Quick, test_entry_threshold_is_inclusive);
    ("exits on reversion", `Quick, test_exits_on_reversion);
    ("stop-loss is direction-aware", `Quick, test_stop_loss_is_direction_aware);
    ("max holding forces an exit", `Quick, test_max_holding_forces_exit);
    ("stop-loss takes precedence over max-holding", `Quick,
     test_stop_loss_takes_precedence_over_max_holding);
    ("an undefined z-score holds", `Quick, test_undefined_zscore_holds);
    ("config rejects exit >= entry", `Quick, test_config_rejects_exit_above_entry);
    ("config rejects stop <= entry", `Quick, test_config_rejects_stop_below_entry);
    ("config rejects bad capital", `Quick, test_config_rejects_bad_capital);
    ("config rejects negative costs", `Quick, test_config_rejects_negative_costs);
    ("default config is valid", `Quick, test_default_config_is_valid);
  ]
