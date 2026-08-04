(** Test suite entry point.

    Suites are ordered from the mechanism outward: causal views, then the
    statistics built on them, then the strategy, then the whole engine, and
    finally the lookahead and property suites that check the guarantees the
    README makes. *)

let () =
  Alcotest.run "statarb"
    [
      ("causal", Test_causal.tests);
      ("rolling", Test_rolling.tests);
      ("ols", Test_ols.tests);
      ("metrics", Test_metrics.tests);
      ("signal", Test_signal.tests);
      ("execution", Test_execution.tests);
      ("csv_io", Test_csv_io.tests);
      ("lookahead", Test_lookahead.tests);
      ("leakage", Test_leakage.tests);
      ("properties", Test_properties.tests);
    ]
