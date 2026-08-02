# v13 fixed-XL feature screen (Phase 3)

Single refit 2023-01-01, fixed 8 epochs, 1 seed — EXPLORATORY (not decision-grade). Control: v12 XL2-on-close_only collapsed to the mean under this exact protocol (S_ref 1.418). Primary question: does richer input prevent XL collapse?

feature_set model  n_features  collapsed  blend_LS_net60  blend_LS_dd   val_ic_best  score_std  train_loss_last  ckpt_mb
     v13_f2 S_ref          27      False           1.570      -0.1730  6.376000e-02    0.11302          0.32089      NaN
     v13_f2    XL          27       True           1.456      -0.1602  1.674400e-01    0.00000          0.33328   1190.3
     v13_f6 S_ref          44      False           1.549      -0.1818  8.674000e-02    0.15958          0.31180      NaN
     v13_f6    XL          44       True           1.456      -0.1602 -1.000000e+09    0.00000          0.33314   1190.6

