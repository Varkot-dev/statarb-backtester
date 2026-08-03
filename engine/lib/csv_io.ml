(** CSV reading and writing: the interface between the OCaml engine and the
    Python layers.

    {1 Schema}

    {2 Input — prices}

    Header: [date,price_a,price_b], one row per bar, strictly increasing dates.

    {[
      date,price_a,price_b
      2020-01-02,100.000000,50.000000
    ]}

    Dates are ISO-8601 [YYYY-MM-DD] and are treated as opaque strings by the
    engine (see {!Types.bar}). Prices must be strictly positive; the engine
    takes logs.

    {2 Output — bars}

    One row per bar with the full audit trail (see {!Types.bar_record}). Every
    number the README reports can be re-derived from this file, which is the
    point: a third party never has to trust the engine's own summary.

    {2 Output — trades, metrics}

    One row per closed trade; and a two-column [metric,value] file.

    {1 Why hand-rolled}

    The parser is deliberately minimal — split on commas, no quoting, no
    embedded newlines. The schema is fully under our control on both sides, so
    a CSV library would be a dependency carrying capability we do not use. The
    parser is strict: any malformed row is a {!Types.Parse_error} carrying its
    line number, never a silently skipped record. Silently skipping rows in a
    backtester is how a data gap becomes an invisible change in the sample. *)

open Types

(** [split_line s] splits on commas and trims surrounding whitespace. *)
let split_line (s : string) : string list =
  String.split_on_char ',' s |> List.map String.trim

(** [read_all_lines path] reads a file into a list of non-empty lines. *)
let read_all_lines (path : string) : (string list, error) result =
  match open_in_bin path with
  | exception Sys_error msg -> Error (Parse_error { line = 0; msg })
  | ic ->
      let rec go acc =
        match input_line ic with
        | line ->
            let t = String.trim line in
            go (if t = "" then acc else t :: acc)
        | exception End_of_file -> List.rev acc
      in
      let lines = go [] in
      close_in ic;
      Ok lines

(** [read_prices path] loads a price CSV into a {!Types.series}.

    Validates: header present and correct; every row has 3 fields; every price
    parses and is strictly positive; dates strictly increasing (so the series
    is genuinely chronological, which every trailing window in the engine
    assumes). *)
let read_prices (path : string) : (series, error) result =
  let open R in
  let* lines = read_all_lines path in
  match lines with
  | [] -> Error (Parse_error { line = 0; msg = "file is empty" })
  | header :: rows ->
      let hs = split_line header in
      let* () =
        match hs with
        | [ "date"; "price_a"; "price_b" ] -> Ok ()
        | _ ->
            Error
              (Parse_error
                 {
                   line = 1;
                   msg =
                     Printf.sprintf
                       "expected header 'date,price_a,price_b', got '%s'" header;
                 })
      in
      let parse_row (lineno : int) (row : string) : (bar, error) result =
        match split_line row with
        | [ date; sa; sb ] ->
            let num name s =
              match float_of_string_opt s with
              | Some x -> Ok x
              | None ->
                  Error
                    (Parse_error
                       { line = lineno; msg = Printf.sprintf "%s: '%s' is not a number" name s })
            in
            let* a = num "price_a" sa in
            let* b = num "price_b" sb in
            let* pa =
              match Price.of_float a with
              | Ok p -> Ok p
              | Error _ ->
                  Error
                    (Parse_error
                       { line = lineno; msg = Printf.sprintf "price_a must be positive and finite, got %g" a })
            in
            let* pb =
              match Price.of_float b with
              | Ok p -> Ok p
              | Error _ ->
                  Error
                    (Parse_error
                       { line = lineno; msg = Printf.sprintf "price_b must be positive and finite, got %g" b })
            in
            if date = "" then
              Error (Parse_error { line = lineno; msg = "date is empty" })
            else Ok { date; price_a = pa; price_b = pb }
        | fields ->
            Error
              (Parse_error
                 {
                   line = lineno;
                   msg =
                     Printf.sprintf "expected 3 fields, got %d"
                       (List.length fields);
                 })
      in
      let* bars =
        List.mapi (fun i r -> parse_row (i + 2) r) rows |> R.all
      in
      let arr = Array.of_list bars in
      let n = Array.length arr in
      if n = 0 then Error (Parse_error { line = 1; msg = "no data rows" })
      else begin
        (* Chronological order is a precondition of every trailing window in
           the engine. Checking it here means the engine itself never has to. *)
        let bad = ref None in
        for i = 1 to n - 1 do
          if !bad = None && String.compare arr.(i).date arr.(i - 1).date <= 0
          then
            bad :=
              Some
                (Parse_error
                   {
                     line = i + 2;
                     msg =
                       Printf.sprintf
                         "dates must be strictly increasing: '%s' does not \
                          follow '%s'"
                         arr.(i).date
                         arr.(i - 1).date;
                   })
        done;
        match !bad with Some e -> Error e | None -> Ok arr
      end

(** [write_lines path lines] writes [lines] with trailing newlines. *)
let write_lines (path : string) (lines : string list) : (unit, error) result =
  match open_out_bin path with
  | exception Sys_error msg -> Error (Parse_error { line = 0; msg })
  | oc ->
      List.iter (fun l -> output_string oc (l ^ "\n")) lines;
      close_out oc;
      Ok ()

(** Render a [float option] as either a fixed-precision number or the empty
    string. Empty (not "nan", not "0") is used for "not computed yet" so that
    pandas reads it as NaN and a reader cannot mistake a warm-up bar for a bar
    where the statistic happened to be zero. *)
let opt_f (x : float option) : string =
  match x with None -> "" | Some v -> Printf.sprintf "%.10f" v

let bar_header =
  "index,date,price_a,price_b,hedge_ratio,spread,zscore,position,qty_a,qty_b,cash,position_value,nav,costs_this_bar,interest_this_bar,trade_event"

let bar_row (r : bar_record) : string =
  Printf.sprintf "%d,%s,%.10f,%.10f,%s,%s,%s,%s,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%s"
    r.r_index r.r_date r.r_price_a r.r_price_b (opt_f r.r_hedge_ratio)
    (opt_f r.r_spread) (opt_f r.r_zscore) r.r_position r.r_qty_a r.r_qty_b
    r.r_cash r.r_position_value r.r_nav r.r_costs_this_bar r.r_interest_this_bar
    r.r_trade_event

let write_bars (path : string) (rows : bar_record list) : (unit, error) result =
  write_lines path (bar_header :: List.map bar_row rows)

let trade_header =
  "direction,entry_index,exit_index,entry_date,exit_date,entry_z,exit_z,pnl_gross,costs,pnl_net,exit_reason,holding_bars"

let trade_row (t : trade) : string =
  Printf.sprintf "%s,%d,%d,%s,%s,%.10f,%.10f,%.10f,%.10f,%.10f,%s,%d"
    (string_of_direction t.t_dir)
    t.entry_index t.exit_index t.entry_date t.exit_date t.t_entry_z t.t_exit_z
    t.pnl_gross t.costs t.pnl_net
    (string_of_exit_reason t.reason)
    t.holding_bars

let write_trades (path : string) (trades : trade list) : (unit, error) result =
  write_lines path (trade_header :: List.map trade_row trades)

let write_metrics (path : string) (rows : (string * string) list) :
    (unit, error) result =
  write_lines path
    ("metric,value" :: List.map (fun (k, v) -> k ^ "," ^ v) rows)
