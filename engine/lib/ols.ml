(** Rolling ordinary least squares for the hedge ratio.

    {1 What the hedge ratio is}

    A pairs trade is only market-neutral if the two legs are sized so their
    price moves offset. If A moves ~beta times as much as B, holding 1 share of
    A against [beta] shares of B leaves a residual (the "spread") that is
    approximately stationary even though A and B individually are not.

    We estimate [beta] by regressing log-price of A on log-price of B over a
    trailing window:

    {[ log P_A(t) = alpha + beta * log P_B(t) + e(t) ]}

    and define the spread as the residual [e(t) = log P_A - alpha - beta * log P_B].

    {1 Why logs}

    Regressing raw prices makes [beta] scale-dependent: the same economic
    relationship gives a different beta if one asset trades at $10 and the
    other at $1000, and the residual's variance drifts as price levels drift.
    In logs, [beta] is an elasticity (a 1% move in B implies a beta% move in A)
    and is invariant to the price level, so a z-score computed on the residual
    is comparable across time. This is also why {!Types.Price} enforces
    positivity — [log] of a non-positive price is not a recoverable condition.

    {1 Why rolling, not full-sample}

    A full-sample regression is lookahead: the beta used to trade in year 1
    would embed price relationships from year 3. The window here ends at [t].

    {1 Why OLS and not Total Least Squares}

    OLS is asymmetric — regressing A on B gives a different beta than B on A,
    because OLS assumes the regressor is measured without error. TLS/orthogonal
    regression is the symmetric alternative and is arguably more correct when
    both legs are noisy. We use OLS because it is what the Engle-Granger
    procedure in the Python pair-selection layer uses, and using a different
    estimator in the engine than in the selection step would mean the pairs
    were selected under one model and traded under another. The asymmetry is
    noted as a limitation in the README. *)

open Types

(** Result of a windowed regression of [y] on [x]. *)
type fit = {
  alpha : float;  (** Intercept. *)
  beta : float;  (** Slope: the hedge ratio. *)
  r_squared : float;  (** Fraction of variance explained; diagnostic only. *)
}

(** Minimum variance in the regressor below which the fit is refused.

    With near-zero variance in [x], [beta = cov / var_x] divides by ~0 and the
    hedge ratio explodes, producing an enormous position in leg B. Refusing the
    fit (and therefore emitting no signal that bar) is the correct response. *)
let min_regressor_variance = 1e-12

(** [fit_window ~y ~x k] regresses the trailing [k] observations of [y] on
    those of [x].

    Both views must be aligned at the same [now]. Uses a two-pass centered
    computation for the same numerical-stability reason as {!Rolling.variance}:
    log-prices are large relative to their dispersion, so the uncentered normal
    equations cancel badly. *)
let fit_window ~(y : float Causal.view) ~(x : float Causal.view) (k : int) :
    (fit, error) result =
  if k < 3 then
    Error
      (Config_error
         (Printf.sprintf "hedge-ratio window must be >= 3, got %d" k))
  else if Causal.now y <> Causal.now x then
    Error (Misaligned_series { left = Causal.now y; right = Causal.now x })
  else
    match (Causal.lookback y k, Causal.lookback x k) with
    | Error e, _ | _, Error e -> Error e
    | Ok ys, Ok xs ->
        let n = float_of_int k in
        let my = Array.fold_left ( +. ) 0. ys /. n in
        let mx = Array.fold_left ( +. ) 0. xs /. n in
        let sxy = ref 0. and sxx = ref 0. and syy = ref 0. in
        for i = 0 to k - 1 do
          let dy = ys.(i) -. my and dx = xs.(i) -. mx in
          sxy := !sxy +. (dx *. dy);
          sxx := !sxx +. (dx *. dx);
          syy := !syy +. (dy *. dy)
        done;
        let var_x = !sxx /. n in
        if var_x < min_regressor_variance then
          Error
            (Degenerate_regression
               (Printf.sprintf "regressor variance %.3e below floor %.3e" var_x
                  min_regressor_variance))
        else begin
          let beta = !sxy /. !sxx in
          let alpha = my -. (beta *. mx) in
          (* R^2 = (cov)^2 / (var_x * var_y). Guard the degenerate case where y
             is constant: the fit is still valid, R^2 is simply undefined, and
             reporting 0 is less misleading than nan propagating into the CSV. *)
          let r_squared =
            if !syy < min_regressor_variance then 0.
            else !sxy *. !sxy /. (!sxx *. !syy)
          in
          Ok { alpha; beta; r_squared }
        end

(** [residual f ~y_t ~x_t] is the spread implied by fit [f] at the current bar:
    [y_t - alpha - beta * x_t].

    Kept separate from {!fit_window} because the engine needs to apply a fit
    estimated at entry time to later bars (to mark the position consistently),
    not only to the bar the fit was estimated on. *)
let residual (f : fit) ~(y_t : float) ~(x_t : float) : float =
  y_t -. f.alpha -. (f.beta *. x_t)
