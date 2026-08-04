(** Kalman filter for a time-varying hedge ratio.

    {1 Why this exists alongside rolling OLS}

    {!Ols.fit_window} estimates the hedge ratio on a trailing window of fixed
    length. That embeds an assumption almost nobody states out loud: that beta
    is {e constant} over the window and then jumps discontinuously as
    observations enter and leave it.

    Neither half of that is true of real pairs. The relationship between two
    companies drifts — one changes its capital structure, its business mix, its
    leverage. And it drifts {e smoothly}, not in steps timed to an arbitrary
    window length.

    Rolling OLS handles this badly in a specific, diagnosable way:

    - {b Window length is a bias/variance dial with no good setting.} Short
      windows track drift but are noisy; long windows are stable but stale.
      There is no length that is right for both.
    - {b Estimates jump when an outlier leaves the window.} A large move
      influences beta fully for [w] bars and then vanishes entirely, producing a
      discontinuity in the hedge ratio that has nothing to do with new
      information.
    - {b It weights a 60-bar-old observation exactly as much as yesterday's.}

    A Kalman filter replaces the window with a {e model}: beta follows a random
    walk, observations are noisy, and the filter computes the optimal (minimum
    mean-squared-error) estimate given everything seen so far. Observations are
    weighted by their informativeness rather than by an arbitrary cutoff, old
    information decays smoothly, and there is no window length to choose.

    {1 The state-space model}

    Observation equation — the spread relationship at bar [t]:

    {v
        y_t = H_t x_t + v_t,     v_t ~ N(0, R)
        H_t = [1, log P_B(t)]           (intercept and hedge ratio regressor)
        x_t = [alpha_t, beta_t]^T       (the hidden state we are estimating)
        y_t = log P_A(t)
    v}

    State equation — the relationship drifts as a random walk:

    {v
        x_t = x_{t-1} + w_t,     w_t ~ N(0, Q)
    v}

    A random walk (rather than a mean-reverting process) is the right prior
    here: we have no view on where beta {e should} be, only that it moves
    slowly. This is the standard formulation for time-varying-parameter
    regression.

    {1 Causality}

    The filter is causal by construction — this is its defining property, not an
    added guarantee. The estimate at bar [t] is conditioned on observations
    [1..t] only, because that is what the recursion computes. There is no smoother
    here (a Kalman {e smoother} conditions on the full sample and would be
    lookahead if traded on); {!step} only ever moves forward.

    That said, the same truncation test applied to everything else in this
    codebase applies here, and [test_kalman.ml] runs it.

    {1 Tuning}

    Only one number really matters: the ratio [Q/R], the {b signal-to-noise
    ratio} of the state process. It is the Kalman analogue of the window length,
    and it trades off exactly the same way:

    - [Q/R] large — beta is believed to move a lot relative to observation
      noise; the filter tracks quickly and is jumpy. (Short window.)
    - [Q/R] small — beta is believed to be nearly constant; the filter is smooth
      and slow. (Long window.)

    The correspondence is approximate but useful: a filter with [Q/R = q] has an
    effective memory of roughly [1/sqrt(q)] bars.

    Reference: Harvey (1989), {i Forecasting, Structural Time Series Models and
    the Kalman Filter}; Chan (2013), {i Algorithmic Trading}, ch. 3, for the
    pairs-trading application. *)

open Types

(** Filter parameters. *)
type params = {
  observation_variance : float;
      (** [R]: variance of the observation noise, i.e. how much of the spread
          is transient noise rather than a genuine change in the relationship. *)
  state_variance : float;
      (** [Q]: per-bar variance of the random walk in [(alpha, beta)]. How fast
          the relationship is believed to drift. *)
  initial_beta : float;
  initial_alpha : float;
  initial_state_variance : float;
      (** Diagonal of the initial state covariance [P_0]. Set large to express
          ignorance about the starting values, so the first observations move
          the estimate quickly (a "diffuse prior"). *)
}

(** Conventional defaults.

    [Q/R = 1e-5] corresponds to an effective memory of roughly 300 bars — a
    slowly-drifting relationship, deliberately more conservative than the
    60-bar rolling OLS default so the two are not merely reparameterisations of
    each other. The comparison in the README sweeps this ratio rather than
    relying on any single choice. *)
let default_params =
  {
    observation_variance = 1e-3;
    state_variance = 1e-8;
    initial_beta = 1.0;
    initial_alpha = 0.0;
    initial_state_variance = 1.0;
  }

let make_params ?(observation_variance = 1e-3) ?(state_variance = 1e-8)
    ?(initial_beta = 1.0) ?(initial_alpha = 0.0)
    ?(initial_state_variance = 1.0) () : (params, error) result =
  if observation_variance <= 0. || not (Float.is_finite observation_variance)
  then Error (Config_error "observation_variance must be positive and finite")
  else if state_variance < 0. || not (Float.is_finite state_variance) then
    Error (Config_error "state_variance must be non-negative and finite")
  else if initial_state_variance <= 0. then
    Error (Config_error "initial_state_variance must be positive")
  else if not (Float.is_finite initial_beta && Float.is_finite initial_alpha)
  then Error (Config_error "initial state must be finite")
  else
    Ok
      {
        observation_variance;
        state_variance;
        initial_beta;
        initial_alpha;
        initial_state_variance;
      }

(** Filter state.

    The 2x2 covariance is stored as three floats rather than a matrix type: it
    is symmetric, so [p01 = p10], and a dedicated matrix module for one fixed
    2x2 case would be more ceremony than the arithmetic it replaces. *)
type state = {
  alpha : float;
  beta : float;
  p00 : float;  (** Var(alpha) *)
  p01 : float;  (** Cov(alpha, beta) *)
  p11 : float;  (** Var(beta) *)
}

let initial (p : params) : state =
  {
    alpha = p.initial_alpha;
    beta = p.initial_beta;
    p00 = p.initial_state_variance;
    p01 = 0.;
    p11 = p.initial_state_variance;
  }

(** Result of processing one observation. *)
type update = {
  new_state : state;
  prediction : float;  (** [E[y_t | data through t-1]] — the forecast. *)
  innovation : float;
      (** [y_t - prediction]. {b This is the spread}: the part of today's
          observation the model did not anticipate. It is the natural signal for
          a pairs trade, and unlike the OLS residual it is a genuine one-step
          prediction error rather than an in-sample fitted residual. *)
  innovation_variance : float;
      (** Variance of the innovation, [S]. Because the filter tracks this, the
          innovation can be standardised by its {e own model-implied} standard
          deviation — no rolling window of past spreads is needed at all. *)
  standardised_innovation : float;
      (** [innovation / sqrt(S)]: a z-score with no window parameter. *)
}

(** [step params state ~log_a ~log_b] processes one observation.

    Standard predict/update recursion. With a random-walk state transition the
    prediction step is just [x] unchanged and [P + Q], which is why no matrix
    multiplication appears.

    Numerical note: [S] is guaranteed positive because [R > 0] is enforced at
    construction and [P] stays positive semi-definite under the Joseph-form-free
    update used here (safe for a 2x2 with a scalar observation). A defensive
    floor is still applied, because a [P] driven to zero by a long run of
    perfectly-predicted observations would otherwise divide by zero. *)
let step (params : params) (s : state) ~(log_a : float) ~(log_b : float) :
    update =
  (* --- Predict: x unchanged, P grows by Q. --- *)
  let p00 = s.p00 +. params.state_variance in
  let p01 = s.p01 in
  let p11 = s.p11 +. params.state_variance in

  (* --- Observation: y = alpha + beta * log_b. --- *)
  let prediction = s.alpha +. (s.beta *. log_b) in
  let innovation = log_a -. prediction in

  (* S = H P H' + R, with H = [1, log_b]. *)
  let ph0 = p00 +. (p01 *. log_b) in
  let ph1 = p01 +. (p11 *. log_b) in
  let innovation_variance =
    p00
    +. (2. *. p01 *. log_b)
    +. (p11 *. log_b *. log_b)
    +. params.observation_variance
  in
  let s_safe = Float.max innovation_variance 1e-15 in

  (* Kalman gain K = P H' / S. *)
  let k0 = ph0 /. s_safe in
  let k1 = ph1 /. s_safe in

  (* --- Update: x += K * innovation, P -= K H P. --- *)
  let alpha = s.alpha +. (k0 *. innovation) in
  let beta = s.beta +. (k1 *. innovation) in
  let new_p00 = p00 -. (k0 *. ph0) in
  let new_p01 = p01 -. (k0 *. ph1) in
  let new_p11 = p11 -. (k1 *. ph1) in

  {
    new_state =
      {
        alpha;
        beta;
        (* Variances cannot go negative; rounding can push them fractionally
           below zero, and letting that propagate would produce a nan gain on
           the next step. *)
        p00 = Float.max new_p00 0.;
        p01 = new_p01;
        p11 = Float.max new_p11 0.;
      };
    prediction;
    innovation;
    innovation_variance = s_safe;
    standardised_innovation = innovation /. sqrt s_safe;
  }

(** [run params series] filters an entire price series, returning one update per
    bar in chronological order.

    Every element of the result at index [t] depends only on bars [0..t]; the
    recursion has no other option. *)
let run (params : params) (series : series) : update array =
  let n = Array.length series in
  let state = ref (initial params) in
  Array.init n (fun i ->
      let bar = series.(i) in
      let u =
        step params !state
          ~log_a:(log (Price.to_float bar.price_a))
          ~log_b:(log (Price.to_float bar.price_b))
      in
      state := u.new_state;
      u)

(** [signal_to_noise params] is [Q/R], the one number that matters for tuning. *)
let signal_to_noise (p : params) : float =
  p.state_variance /. p.observation_variance

(** [effective_memory_bars params] is the approximate rolling-OLS window length
    this filter behaves like, namely [1/sqrt(Q/R)].

    Useful for stating the comparison honestly: a Kalman filter is not
    parameter-free, it just has a parameter that is easier to reason about and
    that degrades gracefully rather than producing window-edge discontinuities. *)
let effective_memory_bars (p : params) : float =
  let q = signal_to_noise p in
  if q <= 0. then infinity else 1. /. sqrt q

(** [warmup_bars params] is a conservative estimate of how many bars the filter
    needs before its estimates are trustworthy.

    With a diffuse prior the first few observations move the state a great deal,
    so early estimates are dominated by the initial guess. Ten bars is ample for
    a two-parameter state; the engine ignores signals before this point. *)
let warmup_bars (_p : params) : int = 10

(** [backtest_with_filter cfg params series] runs the full engine driven by the
    filter's standardised innovation instead of a rolling z-score.

    This exists so the README's memory-law table is {e generated} rather than
    transcribed. That table carries the claim that the window/half-life finding
    generalises beyond windowed estimators, which is the most load-bearing claim
    in the repository; having it live only as prose was the sharpest possible
    contradiction of the project's own thesis that every number is regenerated.

    {2 What signal this trades, and a correction}

    The obvious choice is the filter's {e standardised innovation}, and the
    first version of this function used it. That was wrong, in a way worth
    recording because it is a real conceptual trap.

    The one-step innovation answers {b "how surprised was the model today?"} The
    rolling z-score answers {b "how far from equilibrium is the spread sitting
    right now?"} Mean reversion trades the second. A filter that tracks well has
    {e small} innovations precisely when the spread is most extreme, because it
    has already absorbed the level into its state. Measured on the primary
    dataset, the standardised innovation exceeded |2| on 0 of 2,420 bars at a
    memory ratio of 2 (max |z| = 1.87), so the strategy simply never traded and
    the reported Sharpe was an artifact of an empty trade list.

    What this function trades instead is the {b residual from the filtered
    hedge ratio} — [log P_A - alpha_t - beta_t * log P_B] — standardised over a
    trailing window exactly as {!Backtest.run} standardises the OLS residual.
    That is a level, which is what the strategy needs. The Kalman filter's
    contribution is a better, smoothly-varying beta; it does not replace the
    z-score.

    So the comparison is honest: identical signal definition, identical
    thresholds, identical execution and costs. The only difference from the
    production path is how beta is estimated.

    Causality is inherited rather than re-argued: the filter recursion at bar
    [t] is a function of bars [0..t] by construction, and the intent computed at
    [t] is filled at [t+1] exactly as in {!Backtest.run}. *)
let backtest_with_filter (cfg : Config.t) (params : params) (series : series) :
    (Metrics.t, error) result =
  let open R in
  let n = Array.length series in
  let warm = warmup_bars params in
  if n < warm + 2 then Error (Insufficient_data { needed = warm + 2; got = n })
  else begin
    let filtered = run params series in
    let portfolio = ref (Portfolio.initial cfg) in
    let entry_dates : (int, string) Hashtbl.t = Hashtbl.create 64 in
    let pending = ref Signal.Hold in
    let pending_z = ref 0. in
    let navs = ref [] in
    let total_interest = ref 0. in
    let spread_history = Array.make n 0. in
    let spread_count = ref 0 in
    let abort = ref None in

    let i = ref 0 in
    while !i < n && !abort = None do
      let t = !i in
      let bar = series.(t) in
      let pa = Price.to_float bar.price_a in
      let pb = Price.to_float bar.price_b in

      if t > 0 then begin
        let updated, interest = Portfolio.accrue_interest !portfolio cfg in
        portfolio := updated;
        total_interest := !total_interest +. interest
      end;

      (match !pending with
      | Signal.Hold -> ()
      | intent -> (
          match
            Execution.apply cfg ~intent ~position:!portfolio.position
              ~fill_price_a:pa ~fill_price_b:pb ~fill_index:t
              ~fill_date:bar.date ~beta:filtered.(t).new_state.beta
              ~z:!pending_z ~entry_dates
          with
          | Ok fill -> portfolio := Portfolio.apply_fill !portfolio fill
          | Error e -> abort := Some e));

      if !abort = None then begin
        (* The spread implied by the filtered beta, standardised over a
           trailing window. Grown incrementally, so bar t's spread is computed
           with bar t's beta — the same causal discipline as Backtest.run. *)
        (if t >= warm then begin
           let st = filtered.(t).new_state in
           let spread =
             log (Price.to_float bar.price_a)
             -. st.alpha
             -. (st.beta *. log (Price.to_float bar.price_b))
           in
           spread_history.(!spread_count) <- spread;
           incr spread_count
         end);
        (if t >= warm && !spread_count > cfg.zscore_window then
           let sv = Causal.create spread_history (!spread_count - 1) in
           let z =
             match Rolling.zscore sv cfg.zscore_window with
             | Ok (Some z) -> z
             | _ -> Float.nan
           in
           if Float.is_finite z then begin
             let snapshot =
               {
                 Signal.fit =
                   { Ols.alpha = 0.; beta = filtered.(t).new_state.beta;
                     r_squared = 0. };
                 spread = filtered.(t).innovation;
                 z = Some z;
               }
             in
             pending :=
               Signal.decide ~snapshot ~position:!portfolio.position
                 ~bar_index:t cfg;
             pending_z := z
           end
           else pending := Signal.Hold);
        navs := Portfolio.nav !portfolio ~price_a:pa ~price_b:pb :: !navs
      end;
      incr i
    done;

    match !abort with
    | Some e -> Error e
    | None ->
        let final = series.(n - 1) in
        let fpa = Price.to_float final.price_a in
        let fpb = Price.to_float final.price_b in
        let* final_portfolio =
          match !portfolio.position with
          | Flat -> Ok !portfolio
          | Open _ ->
              let* fill =
                Execution.apply cfg ~intent:(Signal.Exit End_of_data)
                  ~position:!portfolio.position ~fill_price_a:fpa
                  ~fill_price_b:fpb ~fill_index:(n - 1) ~fill_date:final.date
                  ~beta:1. ~z:!pending_z ~entry_dates
              in
              Ok (Portfolio.apply_fill !portfolio fill)
        in
        let final_nav = Portfolio.nav final_portfolio ~price_a:fpa ~price_b:fpb in
        let navs_arr =
          let a = Array.of_list (List.rev !navs) in
          if Array.length a > 0 then a.(Array.length a - 1) <- final_nav;
          a
        in
        Metrics.compute ~navs:navs_arr
          ~trades:(List.rev final_portfolio.trades)
          ~total_costs:final_portfolio.total_costs ~turnover_notional:0.
          ~total_interest:!total_interest ~n_trading_bars:(n - warm)
          ~bars_with_position:0 cfg
  end
