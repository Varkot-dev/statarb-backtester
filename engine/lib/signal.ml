(** Signal generation: hedge ratio, spread, z-score, and the target state.

    This module is pure and causal. It takes views (which cannot see the
    future) and the current position, and returns what the portfolio {e should}
    be. It does not execute anything — {!Execution} turns an intent into fills
    on the following bar. Keeping the decision separate from the fill is what
    makes next-bar execution enforceable rather than aspirational. *)

open Types

(** Everything the engine computed about the spread at one bar.

    [z] is an option because the z-score is undefined when the window standard
    deviation collapses (see {!Rolling.min_stddev}); the engine holds its
    position and emits no new intent in that case. *)
type snapshot = {
  fit : Ols.fit;
  spread : float;
  z : float option;
}

(** What the strategy wants the position to be, decided using data up to and
    including the current bar. *)
type intent =
  | Hold  (** No change. *)
  | Enter of direction
  | Exit of exit_reason
  | Flip of direction
      (** Close the current position and open the opposite one. Represented as
          a single intent rather than an [Exit] followed by an [Enter] so that
          {!Execution} charges the costs of both legs of both trades on the
          same bar and cannot accidentally leave the book flat for a bar. *)

let string_of_intent = function
  | Hold -> "hold"
  | Enter d -> "enter:" ^ string_of_direction d
  | Exit r -> "exit:" ^ string_of_exit_reason r
  | Flip d -> "flip:" ^ string_of_direction d

(** [compute ~log_a ~log_b cfg] produces the spread snapshot at the current bar
    of the two (aligned) log-price views.

    Both views must be at the same [now]. The hedge ratio is re-estimated every
    bar from the trailing [hedge_window]; the spread history over which the
    z-score is computed is supplied by the caller in [spread_view], because it
    must be the series of {e past} spreads, which only the engine loop
    accumulates.

    Note the subtlety in how the spread history is built: at each bar the
    engine appends the spread computed with {e that bar's} hedge ratio. An
    alternative is to recompute the whole spread history with the current
    hedge ratio. The latter is {e not} causal in spirit — it retroactively
    rewrites what the spread "was" at earlier dates using information from
    today — and it makes the z-score jump discontinuously whenever beta moves.
    We use the former. *)
let compute ~(log_a : float Causal.view) ~(log_b : float Causal.view)
    ~(spread_view : float Causal.view option) (cfg : Config.t) :
    (snapshot, error) result =
  match Ols.fit_window ~y:log_a ~x:log_b cfg.hedge_window with
  | Error e -> Error e
  | Ok fit ->
      let spread =
        Ols.residual fit ~y_t:(Causal.current log_a) ~x_t:(Causal.current log_b)
      in
      let z =
        match spread_view with
        | None -> Ok None
        | Some sv -> (
            match Rolling.zscore sv cfg.zscore_window with
            | Error (Insufficient_data _) ->
                (* Not enough spread history yet. Not an error: it is the
                   expected state during warm-up. *)
                Ok None
            | Error e -> Error e
            | Ok z -> Ok z)
      in
      (match z with Error e -> Error e | Ok z -> Ok { fit; spread; z })

(** [decide ~snapshot ~position ~bar_index cfg] is the trading intent implied
    by the current spread state.

    {1 The decision rules}

    When flat:
    - [z <= -entry]: the spread is unusually {e low}. Expect it to rise, so go
      [Long_spread] (long A, short B).
    - [z >= +entry]: the spread is unusually {e high}. Expect it to fall, so go
      [Short_spread].
    - otherwise: stay flat.

    When holding, checked in this precedence order:
    + {b Stop-loss} — [|z|] has widened past [stop_loss_threshold] {e in the
      direction that hurts}. Checked first: a position that has blown through
      its stop must be closed regardless of what any other rule says.
    + {b Max holding} — the position has been open [max_holding_bars]. Checked
      before reversion so that the cap is a hard bound.
    + {b Reversion} — [|z| <= exit_threshold]: the spread has mean-reverted,
      take the profit.
    + otherwise hold.

    {1 Why no flip on threshold crossing}

    A position is never flipped directly from a stop-loss, because the stop
    fires precisely when the model is behaving unexpectedly. Doubling down by
    reversing at that moment is a distinct strategy with different risk, not a
    detail of this one. [Flip] exists in the type because the engine supports
    it and it is exercised in tests, but the default rules never emit it —
    the strategy always passes through flat. This is stated here so a reader
    does not conclude the constructor is dead code. *)
let decide ~(snapshot : snapshot) ~(position : position) ~(bar_index : int)
    (cfg : Config.t) : intent =
  match snapshot.z with
  | None ->
      (* Z-score undefined this bar (degenerate window). Take no new action;
         an existing position is held rather than liquidated, since an
         undefined signal is not evidence to exit. *)
      Hold
  | Some z -> (
      match position with
      | Flat ->
          if z <= -.cfg.entry_threshold then Enter Long_spread
          else if z >= cfg.entry_threshold then Enter Short_spread
          else Hold
      | Open p ->
          let held = bar_index - p.entry_index in
          (* "Adverse" means the spread moved further from the mean in the
             direction that loses money for this position. A Long_spread was
             opened at very negative z and profits as z rises toward 0, so
             adverse movement is z falling further below the stop. *)
          let stopped_out =
            match p.dir with
            | Long_spread -> z <= -.cfg.stop_loss_threshold
            | Short_spread -> z >= cfg.stop_loss_threshold
          in
          if stopped_out then Exit Stop_loss
          else if held >= cfg.max_holding_bars then Exit Max_holding
          else if Float.abs z <= cfg.exit_threshold then Exit Reversion
          else Hold)
