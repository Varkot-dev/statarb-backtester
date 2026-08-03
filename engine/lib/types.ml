(** Core domain types for the pairs-trading backtester.

    Design principle: make invalid states unrepresentable. Where the OCaml type
    system can rule out a class of bug at compile time, we spend the extra
    constructor to do so rather than relying on runtime asserts or on the
    discipline of whoever edits the code next.

    The two states this module is most concerned with eliminating are:

    - A position that is simultaneously long and short. Representing a position
      as two separate quantity fields ([long_qty], [short_qty]) admits the
      nonsensical state where both are non-zero. Here {!position} is a sum type,
      so "long and short at once" has no constructor and cannot be written.

    - A quantity that is negative when the direction is already encoded
      elsewhere. [Long (-5.)] is sign-confusion waiting to happen. {!Qty} is an
      abstract type whose only constructor validates positivity, so a negative
      magnitude cannot enter the system. *)

(* ------------------------------------------------------------------ *)
(* Errors                                                              *)
(* ------------------------------------------------------------------ *)

(** All failure modes in the engine are values, not exceptions. Control flow
    uses [result]; exceptions are reserved for genuine programmer error
    (invariant violations that indicate a bug, not a bad input). *)
type error =
  | Invalid_quantity of float
      (** A magnitude was non-positive or not finite. *)
  | Invalid_price of float  (** A price was non-positive or not finite. *)
  | Insufficient_data of { needed : int; got : int }
      (** A window statistic was requested before enough history existed. *)
  | Misaligned_series of { left : int; right : int }
      (** Two price series had different lengths. *)
  | Parse_error of { line : int; msg : string }  (** CSV/JSON decode failure. *)
  | Config_error of string  (** A configuration value was out of range. *)
  | Degenerate_regression of string
      (** OLS had zero variance in the regressor. *)

let string_of_error = function
  | Invalid_quantity q -> Printf.sprintf "invalid quantity: %g" q
  | Invalid_price p -> Printf.sprintf "invalid price: %g" p
  | Insufficient_data { needed; got } ->
      Printf.sprintf "insufficient data: needed %d observations, got %d" needed
        got
  | Misaligned_series { left; right } ->
      Printf.sprintf "misaligned series: left has %d rows, right has %d" left
        right
  | Parse_error { line; msg } -> Printf.sprintf "parse error on line %d: %s" line msg
  | Config_error msg -> Printf.sprintf "config error: %s" msg
  | Degenerate_regression msg -> Printf.sprintf "degenerate regression: %s" msg

(** Local [result] helpers. We avoid a dependency on a monad library to keep
    the build to a single opam package (dune) plus the test frameworks. *)
module R = struct
  let ( let* ) = Result.bind
  let ( >>| ) r f = Result.map f r

  (** Sequence a list of results into a result of list, short-circuiting on the
      first error. *)
  let all (xs : ('a, 'e) result list) : ('a list, 'e) result =
    let rec go acc = function
      | [] -> Ok (List.rev acc)
      | Ok x :: tl -> go (x :: acc) tl
      | Error e :: _ -> Error e
    in
    go [] xs
end

(* ------------------------------------------------------------------ *)
(* Quantity: a validated positive magnitude                            *)
(* ------------------------------------------------------------------ *)

(** A strictly positive, finite magnitude of shares.

    This is abstract on purpose. Direction lives in {!position}, never in the
    sign of a quantity, which removes an entire family of sign-flip bugs from
    the PnL accounting. *)
module Qty : sig
  type t

  val of_float : float -> (t, error) result
  (** [of_float x] is [Ok] iff [x] is finite and strictly positive. *)

  val to_float : t -> float
  val add : t -> t -> t
  val scale : t -> float -> (t, error) result
  val compare : t -> t -> int
  val pp : Format.formatter -> t -> unit
end = struct
  type t = float

  let of_float x =
    if Float.is_finite x && x > 0. then Ok x else Error (Invalid_quantity x)

  let to_float x = x
  let add a b = a +. b

  let scale q k =
    let x = q *. k in
    if Float.is_finite x && x > 0. then Ok x else Error (Invalid_quantity x)

  let compare = Float.compare
  let pp fmt x = Format.fprintf fmt "%.6f" x
end

(* ------------------------------------------------------------------ *)
(* Price: a validated positive price                                   *)
(* ------------------------------------------------------------------ *)

(** A strictly positive, finite price.

    Positivity is required because the engine takes logs of prices when
    building the spread; a non-positive price is a data error that should be
    caught at ingestion rather than producing a silent [nan] downstream. *)
module Price : sig
  type t

  val of_float : float -> (t, error) result
  val to_float : t -> float
  val pp : Format.formatter -> t -> unit
end = struct
  type t = float

  let of_float x =
    if Float.is_finite x && x > 0. then Ok x else Error (Invalid_price x)

  let to_float x = x
  let pp fmt x = Format.fprintf fmt "%.6f" x
end

(* ------------------------------------------------------------------ *)
(* Position                                                            *)
(* ------------------------------------------------------------------ *)

(** Which leg of the pair a quantity refers to. *)
type leg = Leg_a | Leg_b

(** The direction of a spread position.

    In pairs trading the "spread" is long one asset and short the other. We
    name the two directions by what they do to the spread, which is how the
    signal logic thinks about them:

    - [Long_spread]: buy A, sell B. Entered when the spread is unusually low
      (z-score below the negative entry threshold) and expected to revert up.
    - [Short_spread]: sell A, buy B. Entered when the spread is unusually high.

    Note there is no constructor combining both, and no way to hold two
    positions in the same pair simultaneously. *)
type direction = Long_spread | Short_spread

let opposite = function
  | Long_spread -> Short_spread
  | Short_spread -> Long_spread

let string_of_direction = function
  | Long_spread -> "long_spread"
  | Short_spread -> "short_spread"

(** An open position in a pair.

    [qty_a] and [qty_b] are magnitudes; the signs are implied by [dir]. The
    hedge ratio at entry is retained so that PnL attribution and the exit
    trade use the same ratio the position was opened with — recomputing the
    hedge ratio at exit time would silently change the economics of the trade
    that was actually entered. *)
type open_position = {
  dir : direction;
  qty_a : Qty.t;
  qty_b : Qty.t;
  entry_price_a : Price.t;
  entry_price_b : Price.t;
  entry_index : int;  (** Bar index at which the fill occurred. *)
  entry_hedge_ratio : float;
  entry_z : float;  (** z-score that triggered entry, for diagnostics. *)
}

(** A position is either flat or open. There is no third state, and an open
    position carries exactly one direction. *)
type position = Flat | Open of open_position

let is_flat = function Flat -> true | Open _ -> false

(** Signed exposure in leg [l], in shares. This is the single place where a
    direction is turned into a sign; every other module asks this function
    rather than reimplementing the convention. *)
let signed_qty (p : open_position) (l : leg) : float =
  let mag = match l with Leg_a -> Qty.to_float p.qty_a | Leg_b -> Qty.to_float p.qty_b in
  match (p.dir, l) with
  | Long_spread, Leg_a -> mag (* long A *)
  | Long_spread, Leg_b -> -.mag (* short B *)
  | Short_spread, Leg_a -> -.mag (* short A *)
  | Short_spread, Leg_b -> mag (* long B *)

(* ------------------------------------------------------------------ *)
(* Bars and series                                                     *)
(* ------------------------------------------------------------------ *)

(** One observation of the pair. Dates are kept as ISO-8601 strings: the engine
    never does date arithmetic (annualization uses a bars-per-year constant),
    so a real date type would be unused ceremony. *)
type bar = { date : string; price_a : Price.t; price_b : Price.t }

(** A price series is an immutable array of bars, indexed 0..n-1 in
    chronological order. Chronological ordering is a precondition checked at
    load time in {!Csv_io}. *)
type series = bar array

(* ------------------------------------------------------------------ *)
(* Trades                                                              *)
(* ------------------------------------------------------------------ *)

(** Why a position was closed. Kept distinct because the README reports the
    breakdown, and because a strategy that exits mostly on stops behaves very
    differently from one that exits on mean reversion. *)
type exit_reason =
  | Reversion  (** |z| fell below the exit threshold. *)
  | Stop_loss  (** |z| widened past the stop threshold. *)
  | Max_holding  (** Position hit the maximum holding period. *)
  | End_of_data  (** Backtest ended with the position still open. *)

let string_of_exit_reason = function
  | Reversion -> "reversion"
  | Stop_loss -> "stop_loss"
  | Max_holding -> "max_holding"
  | End_of_data -> "end_of_data"

(** A completed round-trip trade. [pnl_net] is after all costs. *)
type trade = {
  t_dir : direction;
  entry_index : int;
  exit_index : int;
  entry_date : string;
  exit_date : string;
  t_entry_z : float;
  t_exit_z : float;
  pnl_gross : float;
  costs : float;  (** Commission + slippage, both legs, both sides. *)
  pnl_net : float;
  reason : exit_reason;
  holding_bars : int;
}

(* ------------------------------------------------------------------ *)
(* Per-bar diagnostic record                                           *)
(* ------------------------------------------------------------------ *)

(** One row of the backtest audit trail.

    Every field here is emitted to CSV so that the Python reporting layer, and
    any third party, can re-derive the reported metrics from first principles
    rather than trusting the engine's own summary. [nav] is the single source
    of truth for performance: metrics are computed from the NAV series, not
    accumulated separately. *)
type bar_record = {
  r_index : int;
  r_date : string;
  r_price_a : float;
  r_price_b : float;
  r_hedge_ratio : float option;
  r_spread : float option;
  r_zscore : float option;
  r_position : string;  (** "flat" | "long_spread" | "short_spread" *)
  r_qty_a : float;  (** Signed. *)
  r_qty_b : float;  (** Signed. *)
  r_cash : float;
  r_position_value : float;
  r_nav : float;
  r_costs_this_bar : float;
  r_interest_this_bar : float;  (** Risk-free interest credited on cash. *)
  r_trade_event : string;  (** "" | "entry" | "exit:<reason>" | "flip" *)
}
