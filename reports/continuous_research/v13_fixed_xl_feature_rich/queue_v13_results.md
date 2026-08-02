# v13 small-model feature screen (Phase 2)

Bar = close_only (F0) blend LS net60 **1.969** (v12 Wall_baseline reuse; same protocol/cache/code). Reject rules (Task 6): material Sharpe drop, worse DD, turnover explosion, overlap collapse without return gain, val-IC-only wins. Prior: richer sets were REJECTED at this model size in v1-v7 — this screen is also a replication control; the XL hypothesis is tested in Phase 3.

                  id phase feature_set status  n_features  val_ic  blend_LS_net60  blend_LS_dd  blend_turnover  tf_LS_net60  train_s error  q5_overlap_vs_close_only
P2_screen_close_only    P2  close_only   done          10 0.04813           1.969      -0.1054        0.301402        1.656   1604.3                             NaN
    P2_screen_v13_f1    P2      v13_f1   done          13 0.04938           1.420      -0.2083        0.448288        0.752   2097.2                           0.241
    P2_screen_v13_f2    P2      v13_f2   done          27 0.06411           2.101      -0.0839        0.420907        1.225   1927.2                           0.302
    P2_screen_v13_f3    P2      v13_f3   done          32 0.06737           1.447      -0.2481        0.389867        1.041   1255.5                           0.324
    P2_screen_v13_f4    P2      v13_f4   done          37 0.06200           1.523      -0.1819        0.378708        1.113   1017.3                           0.357
    P2_screen_v13_f5    P2      v13_f5   done          40 0.07075           1.592      -0.2138        0.419541        0.772   1183.0                           0.343
    P2_screen_v13_f6    P2      v13_f6   done          44 0.06446           1.547      -0.2235        0.392514        0.730   1171.6                           0.345
    P3_xl_close_only    P3  close_only   done          10     NaN             NaN          NaN             NaN          NaN      NaN                             NaN
  P3_xl_BEST_FROM_P2    P3      v13_f2   done          27     NaN             NaN          NaN             NaN          NaN      NaN                           0.302
        P3_xl_v13_f6    P3      v13_f6   done          44     NaN             NaN          NaN             NaN          NaN      NaN                           0.345
    P2B_v13_f2_CH_7s   P2B      v13_f2   done          27 0.06907           1.601      -0.1293        0.422895        1.403   3085.6                           0.302
    P2B_v13_f2_BR_7s   P2B      v13_f2   done          27 0.07411           1.011      -0.2262        0.443182        0.723   3443.5                           0.302
   P2B_v13_f2_CH_bat   P2B      v13_f2   done          27 0.06480           1.682      -0.1622        0.439285        1.005   2255.3                           0.302
   P2B_v13_f2_BR_bat   P2B      v13_f2   done          27 0.07486           1.076      -0.2590        0.451963        0.750   3154.6                           0.302

