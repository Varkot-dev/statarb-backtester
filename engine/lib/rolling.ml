(** Trailing-window statistics.

    Every function here takes a {!Causal.view}, never a raw array. That is the
    enforcement mechanism for the no-lookahead guarantee: a view cannot address
    indices beyond [now], so a statistic computed from one is necessarily a
    function of data at or before [now].

    All windows are {e trailing}: the window ending at [t] spans
    [t - k + 1 .. t] inclusive. There are no centered windows in this codebase,
    and adding one would require bypassing this module entirely. *)

open Types

(** [mean v k] is the arithmetic mean of the trailing [k] observations.

    Errors with [Insufficient_data] when fewer than [k] observations are
    visible, rather than averaging a shorter window. *)
let mean (v : float Causal.view) (k : int) : (float, error) result =
  match Causal.fold_lookback v k ~init:0. ~f:( +. ) with
  | Error e -> Error e
  | Ok total -> Ok (total /. float_of_int k)

(** [variance v k] is the {e sample} variance (Bessel-corrected, denominator
    [k-1]) of the trailing [k] observations.

    The sample correction matters for the z-score: with [k = 60] the difference
    between dividing by 59 and 60 is ~0.8% in the standard deviation, which
    shifts the effective entry threshold. We use the sample estimator because
    the window is a sample of an unknown process, not the population.

    Computed with a two-pass algorithm. The one-pass sum-of-squares form
    ([E[x^2] - E[x]^2]) is faster but catastrophically cancels when the mean is
    large relative to the spread of the data — which is exactly the regime here,
    since log-price spreads hover near a non-zero level with small variance.
    Two passes over a window of 60 is not a bottleneck. *)
let variance (v : float Causal.view) (k : int) : (float, error) result =
  if k < 2 then
    Error (Config_error (Printf.sprintf "variance needs window >= 2, got %d" k))
  else
    match mean v k with
    | Error e -> Error e
    | Ok mu -> (
        match
          Causal.fold_lookback v k ~init:0. ~f:(fun acc x ->
              let d = x -. mu in
              acc +. (d *. d))
        with
        | Error e -> Error e
        | Ok ss -> Ok (ss /. float_of_int (k - 1)))

(** [stddev v k] is the sample standard deviation of the trailing [k]
    observations. *)
let stddev (v : float Causal.view) (k : int) : (float, error) result =
  match variance v k with Error e -> Error e | Ok var -> Ok (sqrt var)

(** Minimum standard deviation below which a z-score is treated as undefined.

    Without this guard, a window in which the spread is (numerically) constant
    produces a division by ~0 and a z-score of ±1e15, which trips any entry
    threshold and generates a spurious trade. The floor is expressed in the
    units of the spread (log-price), where 1e-10 is far below any real price
    movement. *)
let min_stddev = 1e-10

(** [zscore v k] is [(x_t - mean) / stddev] over the trailing [k] observations,
    where [x_t] is the current value {e and is itself included in the window}.

    Including the current observation is the standard convention and is causal:
    [x_t] is known at time [t]. The alternative (standardising [x_t] against a
    window ending at [t-1]) is also causal and is arguably cleaner, but it is
    not what practitioners mean by "rolling z-score", so we follow convention
    and document it.

    Returns [None] when the window standard deviation is below {!min_stddev},
    signalling "no usable signal this bar" rather than fabricating one. *)
let zscore (v : float Causal.view) (k : int) : (float option, error) result =
  match mean v k with
  | Error e -> Error e
  | Ok mu -> (
      match stddev v k with
      | Error e -> Error e
      | Ok sd ->
          if sd < min_stddev then Ok None
          else Ok (Some ((Causal.current v -. mu) /. sd)))

(** [correlation va vb k] is the Pearson correlation of the trailing [k]
    observations of two views.

    Both views must be at the same [now]; a mismatch means the caller has
    advanced one series without the other, which would silently correlate
    misaligned data. *)
let correlation (va : float Causal.view) (vb : float Causal.view) (k : int) :
    (float option, error) result =
  if Causal.now va <> Causal.now vb then
    Error
      (Misaligned_series { left = Causal.now va; right = Causal.now vb })
  else
    match (Causal.lookback va k, Causal.lookback vb k) with
    | Error e, _ | _, Error e -> Error e
    | Ok xs, Ok ys ->
        let n = float_of_int k in
        let mx = Array.fold_left ( +. ) 0. xs /. n in
        let my = Array.fold_left ( +. ) 0. ys /. n in
        let sxy = ref 0. and sxx = ref 0. and syy = ref 0. in
        for i = 0 to k - 1 do
          let dx = xs.(i) -. mx and dy = ys.(i) -. my in
          sxy := !sxy +. (dx *. dy);
          sxx := !sxx +. (dx *. dx);
          syy := !syy +. (dy *. dy)
        done;
        let denom = sqrt (!sxx *. !syy) in
        if denom < min_stddev then Ok None else Ok (Some (!sxy /. denom))

(** [simple_returns navs] converts a NAV series to simple period returns,
    [r_t = nav_t / nav_{t-1} - 1], length [n-1].

    Simple (not log) returns are used throughout because the Sharpe ratio and
    drawdown conventions the README reports are defined on simple returns, and
    mixing the two is a common source of small but real discrepancies. *)
let simple_returns (navs : float array) : (float array, error) result =
  let n = Array.length navs in
  if n < 2 then Error (Insufficient_data { needed = 2; got = n })
  else begin
    let out = Array.make (n - 1) 0. in
    let bad = ref None in
    for i = 1 to n - 1 do
      let prev = navs.(i - 1) in
      if prev <= 0. then
        if !bad = None then bad := Some (Invalid_price prev) else ()
      else out.(i - 1) <- (navs.(i) /. prev) -. 1.
    done;
    match !bad with Some e -> Error e | None -> Ok out
  end
