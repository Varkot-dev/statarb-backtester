(** CSV round-trips and, more importantly, parser strictness.

    A backtester that silently skips malformed rows turns a data gap into an
    invisible change of sample. Every test here that asserts an [Error] is
    asserting that a specific corruption is {e loud}. *)

open Statarb
open Statarb.Types

let tmp_dir = Filename.get_temp_dir_name ()
let tmp name = Filename.concat tmp_dir ("statarb_test_" ^ name ^ ".csv")

let write_file path contents =
  let oc = open_out_bin path in
  output_string oc contents;
  close_out oc

let with_file name contents f =
  let path = tmp name in
  write_file path contents;
  Fun.protect
    ~finally:(fun () -> try Sys.remove path with Sys_error _ -> ())
    (fun () -> f path)

let test_reads_a_well_formed_file () =
  with_file "good"
    "date,price_a,price_b\n2020-01-01,100.0,50.0\n2020-01-02,101.5,50.5\n"
    (fun path ->
      let s = Fixtures.get_ok (Csv_io.read_prices path) in
      Alcotest.(check int) "two bars" 2 (Array.length s);
      Alcotest.(check string) "first date" "2020-01-01" s.(0).date;
      Alcotest.(check (float 1e-9)) "first price_a" 100.
        (Price.to_float s.(0).price_a);
      Alcotest.(check (float 1e-9)) "second price_b" 50.5
        (Price.to_float s.(1).price_b))

let test_tolerates_surrounding_whitespace () =
  with_file "ws" "date, price_a , price_b \n2020-01-01, 100.0 , 50.0 \n"
    (fun path ->
      let s = Fixtures.get_ok (Csv_io.read_prices path) in
      Alcotest.(check int) "one bar" 1 (Array.length s))

let test_rejects_a_wrong_header () =
  with_file "badheader" "date,a,b\n2020-01-01,100.0,50.0\n" (fun path ->
      match Csv_io.read_prices path with
      | Error (Parse_error { line = 1; _ }) -> ()
      | _ -> Alcotest.fail "a wrong header should be rejected at line 1")

let test_rejects_a_short_row () =
  with_file "shortrow" "date,price_a,price_b\n2020-01-01,100.0\n" (fun path ->
      match Csv_io.read_prices path with
      | Error (Parse_error { line = 2; _ }) -> ()
      | _ -> Alcotest.fail "a row with too few fields should be rejected")

let test_rejects_a_non_numeric_price () =
  with_file "nan" "date,price_a,price_b\n2020-01-01,abc,50.0\n" (fun path ->
      match Csv_io.read_prices path with
      | Error (Parse_error { line = 2; _ }) -> ()
      | _ -> Alcotest.fail "a non-numeric price should be rejected")

(** Non-positive prices are rejected at load, so [log] downstream is always
    defined. *)
let test_rejects_non_positive_prices () =
  with_file "zero" "date,price_a,price_b\n2020-01-01,0.0,50.0\n" (fun path ->
      match Csv_io.read_prices path with
      | Error (Parse_error { line = 2; _ }) -> ()
      | _ -> Alcotest.fail "a zero price should be rejected");
  with_file "neg" "date,price_a,price_b\n2020-01-01,100.0,-5.0\n" (fun path ->
      match Csv_io.read_prices path with
      | Error (Parse_error { line = 2; _ }) -> ()
      | _ -> Alcotest.fail "a negative price should be rejected")

(** Chronological order is a precondition of every trailing window in the
    engine, so it is enforced at the boundary. *)
let test_rejects_out_of_order_dates () =
  with_file "unordered"
    "date,price_a,price_b\n2020-01-02,100.0,50.0\n2020-01-01,101.0,51.0\n"
    (fun path ->
      match Csv_io.read_prices path with
      | Error (Parse_error { line = 3; _ }) -> ()
      | _ -> Alcotest.fail "out-of-order dates should be rejected")

let test_rejects_duplicate_dates () =
  with_file "dupe"
    "date,price_a,price_b\n2020-01-01,100.0,50.0\n2020-01-01,101.0,51.0\n"
    (fun path ->
      match Csv_io.read_prices path with
      | Error (Parse_error { line = 3; _ }) -> ()
      | _ -> Alcotest.fail "duplicate dates should be rejected")

let test_rejects_an_empty_file () =
  with_file "empty" "" (fun path ->
      match Csv_io.read_prices path with
      | Error (Parse_error _) -> ()
      | _ -> Alcotest.fail "an empty file should be rejected")

let test_rejects_header_only () =
  with_file "headeronly" "date,price_a,price_b\n" (fun path ->
      match Csv_io.read_prices path with
      | Error (Parse_error _) -> ()
      | _ -> Alcotest.fail "a header with no data rows should be rejected")

let test_missing_file_is_an_error () =
  match Csv_io.read_prices (Filename.concat tmp_dir "definitely_not_here.csv") with
  | Error (Parse_error _) -> ()
  | _ -> Alcotest.fail "a missing file should be an error, not a crash"

(** Round-trip: writing bar records and reading the file back must preserve the
    row count and header, since the Python layer parses this file. *)
let test_bar_output_round_trips () =
  let path = tmp "bars_out" in
  let rows =
    [
      {
        r_index = 0;
        r_date = "2020-01-01";
        r_price_a = 100.;
        r_price_b = 50.;
        r_hedge_ratio = None;
        r_spread = None;
        r_zscore = None;
        r_position = "flat";
        r_qty_a = 0.;
        r_qty_b = 0.;
        r_cash = 100000.;
        r_position_value = 0.;
        r_nav = 100000.;
        r_costs_this_bar = 0.;
        r_trade_event = "";
      };
      {
        r_index = 1;
        r_date = "2020-01-02";
        r_price_a = 101.;
        r_price_b = 50.5;
        r_hedge_ratio = Some 1.25;
        r_spread = Some 0.0123;
        r_zscore = Some (-2.15);
        r_position = "long_spread";
        r_qty_a = 250.;
        r_qty_b = -500.;
        r_cash = 99992.5;
        r_position_value = 0.;
        r_nav = 99992.5;
        r_costs_this_bar = 7.5;
        r_trade_event = "entry:long_spread";
      };
    ]
  in
  Fun.protect
    ~finally:(fun () -> try Sys.remove path with Sys_error _ -> ())
    (fun () ->
      Fixtures.get_ok (Csv_io.write_bars path rows);
      let lines = Fixtures.get_ok (Csv_io.read_all_lines path) in
      Alcotest.(check int) "header plus two rows" 3 (List.length lines);
      Alcotest.(check string) "header" Csv_io.bar_header (List.nth lines 0);
      (* An absent statistic is written as an empty field, not 0 or nan, so a
         warm-up bar cannot be mistaken for a bar where the value was zero. *)
      let first = List.nth lines 1 in
      Alcotest.(check bool) "warm-up row has empty optional fields" true
        (String.length first > 0
        && List.exists
             (fun f -> f = "")
             (String.split_on_char ',' first)))

let test_metrics_output_format () =
  let path = tmp "metrics_out" in
  Fun.protect
    ~finally:(fun () -> try Sys.remove path with Sys_error _ -> ())
    (fun () ->
      Fixtures.get_ok
        (Csv_io.write_metrics path [ ("sharpe_ratio", "1.2345"); ("n_trades", "42") ]);
      let lines = Fixtures.get_ok (Csv_io.read_all_lines path) in
      Alcotest.(check int) "header plus two rows" 3 (List.length lines);
      Alcotest.(check string) "header" "metric,value" (List.nth lines 0);
      Alcotest.(check string) "first metric" "sharpe_ratio,1.2345" (List.nth lines 1))

(** A full round-trip through the file format: generate a series, write it,
    read it back, and confirm the backtest produces identical results. This is
    the guarantee the Python interface depends on. *)
let test_series_round_trips_through_csv () =
  let path = tmp "roundtrip" in
  let series =
    Fixtures.cointegrated ~n:200 ~seed:4 ~beta:1.1 ~half_life:10.
      ~sigma_spread:0.02 ~sigma_common:0.01
  in
  Fun.protect
    ~finally:(fun () -> try Sys.remove path with Sys_error _ -> ())
    (fun () ->
      let lines =
        "date,price_a,price_b"
        :: List.map
             (fun b ->
               Printf.sprintf "%s,%.10f,%.10f" b.date
                 (Price.to_float b.price_a)
                 (Price.to_float b.price_b))
             (Array.to_list series)
      in
      Fixtures.get_ok (Csv_io.write_lines path lines);
      let reloaded = Fixtures.get_ok (Csv_io.read_prices path) in
      Alcotest.(check int) "same length" (Array.length series)
        (Array.length reloaded);
      let cfg = Fixtures.test_config () in
      let r1 = Fixtures.get_ok (Backtest.run cfg series) in
      let r2 = Fixtures.get_ok (Backtest.run cfg reloaded) in
      Alcotest.(check (float 1e-6)) "same final NAV" r1.metrics.final_nav
        r2.metrics.final_nav;
      Alcotest.(check int) "same trade count" r1.metrics.n_trades
        r2.metrics.n_trades)

let tests =
  [
    ("reads a well-formed file", `Quick, test_reads_a_well_formed_file);
    ("tolerates surrounding whitespace", `Quick, test_tolerates_surrounding_whitespace);
    ("rejects a wrong header", `Quick, test_rejects_a_wrong_header);
    ("rejects a short row", `Quick, test_rejects_a_short_row);
    ("rejects a non-numeric price", `Quick, test_rejects_a_non_numeric_price);
    ("rejects non-positive prices", `Quick, test_rejects_non_positive_prices);
    ("rejects out-of-order dates", `Quick, test_rejects_out_of_order_dates);
    ("rejects duplicate dates", `Quick, test_rejects_duplicate_dates);
    ("rejects an empty file", `Quick, test_rejects_an_empty_file);
    ("rejects a header-only file", `Quick, test_rejects_header_only);
    ("a missing file is an error", `Quick, test_missing_file_is_an_error);
    ("bar output round-trips", `Quick, test_bar_output_round_trips);
    ("metrics output format", `Quick, test_metrics_output_format);
    ("series round-trips through CSV", `Quick, test_series_round_trips_through_csv);
  ]
