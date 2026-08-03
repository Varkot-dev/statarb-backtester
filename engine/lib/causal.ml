(** Causal (non-anticipating) access to a time series.

    {1 Why this module exists}

    Lookahead bias is the failure mode that makes a backtest worthless. It is
    easy to introduce by accident and hard to spot by reading code, because the
    offending expression usually looks innocuous:

    {[
      (* All three of these are lookahead. *)
      let mu = mean whole_series in           (* full-sample mean *)
      let w = Array.sub xs (t - k) (2*k + 1)  (* centered window *)
      let next = xs.(t + 1) in                (* peeking forward *)
    ]}

    Rather than hoping code review catches these, this module makes them
    impossible to write. A {!view} is a handle onto a series that is
    {e physically incapable} of reading index > [t]. Every rolling statistic in
    {!Rolling} takes a [view] instead of an array, so the signal pipeline has
    no access to future data at all — not "does not use it", but "cannot
    address it".

    {1 The guarantee}

    For a view [v] with [now v = t] over an underlying array [xs]:

    - [get v i] returns [Some xs.(i)] for [0 <= i <= t], and [None] otherwise.
    - Therefore any function of [v] alone is a function of [xs.(0..t)] alone.

    That second point is the whole argument. A statistic computed from a view
    is, by construction, a function only of data at or before [t]. This is what
    [test/test_lookahead.ml] verifies empirically by truncating the underlying
    array after [t] and asserting every signal is bit-identical.

    {1 Cost}

    A view is O(1) to create and carries no copy of the data — it is a record
    of an array reference plus an integer bound. Advancing is allocation-free
    in the sense that it produces a small immutable record; the underlying
    array is shared. *)

open Types

(** A non-anticipating window onto [data], exposing indices [0 .. now].

    The fields are exposed for pattern matching within the library but the
    accessors below are the intended interface; nothing outside this module
    reads [data] directly. *)
type 'a view = { data : 'a array; now : int }

(** [create xs t] is a view exposing [xs.(0..t)].

    @raise Invalid_argument if [t] is outside [0, length xs - 1]. This is a
    programmer error (the caller controls [t] via {!advance}), not a data
    condition, so it is an exception rather than a [result]. *)
let create (data : 'a array) (now : int) : 'a view =
  if now < 0 || now >= Array.length data then
    invalid_arg
      (Printf.sprintf "Causal.create: now=%d out of bounds for length %d" now
         (Array.length data));
  { data; now }

(** [now v] is the index of the most recent observation visible through [v]. *)
let now (v : 'a view) : int = v.now

(** [length v] is the number of visible observations, i.e. [now v + 1]. *)
let length (v : 'a view) : int = v.now + 1

(** [advance v] moves the view forward one bar, or returns [None] at the end of
    the underlying data. This is the only way to see more data, and it only
    moves forward — there is no [rewind] that could be misused to reconstruct a
    future-aware window. *)
let advance (v : 'a view) : 'a view option =
  if v.now + 1 < Array.length v.data then Some { v with now = v.now + 1 }
  else None

(** [get v i] is [Some v.data.(i)] when [0 <= i <= now v], else [None].

    The [option] is deliberate. An out-of-range read is not an error to be
    logged and ignored; it is a signal that the caller's window logic is wrong,
    and forcing the caller to handle [None] means the mistake surfaces at the
    call site instead of silently reading garbage. *)
let get (v : 'a view) (i : int) : 'a option =
  if i >= 0 && i <= v.now then Some v.data.(i) else None

(** [current v] is the observation at [now v]. Always present, since a view
    cannot be constructed with [now < 0]. *)
let current (v : 'a view) : 'a = v.data.(v.now)

(** [lookback v k] is the most recent [k] observations, oldest first, i.e.
    indices [now - k + 1 .. now].

    Returns [Error (Insufficient_data _)] rather than a short window when fewer
    than [k] observations exist. Silently returning a shorter window is how
    warm-up periods produce statistics with the wrong denominator; an explicit
    error forces the caller to decide what to do before enough history exists
    (the engine emits no signal at all). *)
let lookback (v : 'a view) (k : int) : ('a array, error) result =
  if k <= 0 then Error (Config_error (Printf.sprintf "lookback window must be positive, got %d" k))
  else
    let avail = length v in
    if avail < k then Error (Insufficient_data { needed = k; got = avail })
    else Ok (Array.sub v.data (v.now - k + 1) k)

(** [fold_lookback v k ~init ~f] folds [f] over the most recent [k] values,
    oldest first, without materialising the window.

    Used by the rolling statistics on the hot path so that a backtest over
    n bars with window k does not allocate n arrays of size k. *)
let fold_lookback (v : 'a view) (k : int) ~(init : 'acc)
    ~(f : 'acc -> 'a -> 'acc) : ('acc, error) result =
  if k <= 0 then Error (Config_error (Printf.sprintf "lookback window must be positive, got %d" k))
  else
    let avail = length v in
    if avail < k then Error (Insufficient_data { needed = k; got = avail })
    else begin
      let acc = ref init in
      for i = v.now - k + 1 to v.now do
        acc := f !acc v.data.(i)
      done;
      Ok !acc
    end

(** [to_visible_array v] copies out everything visible. Intended for tests and
    for the truncation argument in the lookahead test; the engine itself uses
    {!fold_lookback}. *)
let to_visible_array (v : 'a view) : 'a array = Array.sub v.data 0 (length v)
