(** Causal (non-anticipating) access to a time series.

    {1 Why this interface file exists}

    This [.mli] is the whole guarantee. Without it, [type 'a view = { data : 'a
    array; now : int }] is a public record, and every one of these compiles:

    {[
      v.data.(t + 100)                  (* read the field directly *)
      Causal.create v.data (n - 1)      (* "create" is a rewind in disguise *)
      { Causal.data = xs; now = n - 1 } (* forge a view with any bound *)
    ]}

    With [view] abstract, none of them typecheck. [data] and [now] are not
    field names outside this module — they do not exist as names at all — so a
    caller holding a [view] has exactly the operations listed below and no way
    to reach the underlying array or to construct a view over data it was not
    handed.

    That distinction is the difference between a convention and a guarantee, and
    it was originally missing here: the implementation carried a comment saying
    "nothing outside this module reads [data] directly", which is precisely the
    code-review promise this design exists to replace. [test_causal_abstraction.ml]
    now pins the property so it cannot regress.

    {1 The guarantee}

    For a view [v] with [now v = t] over an underlying series [xs]:

    - [get v i] is [Some xs.(i)] for [0 <= i <= t], and [None] otherwise.
    - There is no operation that returns [xs] itself.
    - There is no operation that increases [now] except {!advance}, which moves
      forward exactly one bar and cannot skip.

    Therefore {b any function whose only access to the data is through a view is
    a function of [xs.(0..t)] alone}. That is a property of the type, not of the
    caller's discipline, and it is what makes the truncation-invariance results
    in [test_lookahead.ml] a consequence rather than a coincidence.

    {1 Constructing a view}

    {!create} takes a raw array, and is therefore the one place where the bound
    is chosen rather than enforced. It is the trusted boundary of the module:
    the engine calls it once per bar with the loop index, and everything
    downstream is bounded by that choice. Keeping the trusted surface to a single
    named function is the point — it is auditable by reading one call site.

    {!Leakage} deliberately does not use views at all. It takes raw arrays,
    because injecting a controlled dose of lookahead is its job. That asymmetry
    is the tell: {b a function taking a view is causal by construction; a
    function taking an array is under suspicion.} *)

type 'a view
(** A non-anticipating window onto a series, exposing indices [0 .. now].

    Abstract: the underlying array and the bound are not reachable from outside
    this module. *)

val create : 'a array -> int -> 'a view
(** [create xs t] is a view exposing [xs.(0..t)].

    The trusted boundary. Callers choose [t]; the engine passes its loop index,
    so the bound advances one bar at a time and never runs ahead of the event
    loop.

    @raise Invalid_argument if [t] is outside [0, length xs - 1]. This is a
    programmer error rather than a data condition, so it raises rather than
    returning a [result]. *)

val now : 'a view -> int
(** The index of the most recent visible observation. *)

val length : 'a view -> int
(** Number of visible observations, i.e. [now v + 1]. *)

val advance : 'a view -> 'a view option
(** [advance v] moves the view forward exactly one bar, or [None] at the end of
    the underlying data.

    The only operation that increases [now]. It moves forward only and one step
    at a time, so a view cannot be walked backwards or jumped to an arbitrary
    bound — which, combined with abstraction, is why no sequence of public calls
    reconstructs a future-aware window. *)

val get : 'a view -> int -> 'a option
(** [get v i] is [Some x] when [0 <= i <= now v], else [None].

    The [option] is deliberate. An out-of-range read is not a condition to log
    and ignore; it means the caller's window arithmetic is wrong, and forcing
    the [None] to be handled surfaces that at the call site. *)

val current : 'a view -> 'a
(** The observation at [now v]. Always present, since a view cannot be
    constructed with [now < 0]. *)

val lookback : 'a view -> int -> ('a array, Types.error) result
(** [lookback v k] is the most recent [k] observations, oldest first, spanning
    [now - k + 1 .. now].

    Returns [Error (Insufficient_data _)] rather than a short window when fewer
    than [k] observations exist. Silently returning a shorter window is how
    warm-up periods produce statistics with the wrong denominator. *)

val fold_lookback :
  'a view -> int -> init:'acc -> f:('acc -> 'a -> 'acc) -> ('acc, Types.error) result
(** [fold_lookback v k ~init ~f] folds over the most recent [k] values, oldest
    first, without materialising the window.

    Used on the hot path so that a backtest over n bars with window k does not
    allocate n arrays of size k. *)

val to_visible_array : 'a view -> 'a array
(** A {e copy} of everything currently visible, i.e. [xs.(0 .. now v)].

    This is the only way to obtain an array from a view, and it deliberately
    cannot leak the future: it copies exactly the visible prefix, so the result
    is bounded by the same [now] as the view it came from. Intended for tests
    and for the truncation argument; the engine itself uses {!fold_lookback}. *)
