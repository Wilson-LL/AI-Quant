# v12 recency / training-window comparison (3 seeds, CH window)

Bar = Wall_baseline (equal-weight all-history, the production recipe). Pre-registered reject rules: val-IC-only wins, worse book/bear/turnover, unstable books (plan §5).

       recipe status  val_ic  blend_LS_net60  blend_LS_dd  blend_LO_net60  tf_LS_net60 blend_2022  turnover  train_s  q5_overlap_vs_baseline
Wall_baseline   done 0.04813           1.969      -0.1054           1.937        1.656       None  0.301402   1604.3                     NaN
          W1y   done 0.02593           0.381      -0.2989           0.975       -0.315       None  0.404064    114.4                   0.405
          W2y   done 0.02276           0.588      -0.2553           1.335       -0.631       None  0.451625    360.1                   0.366
          W3y   done 0.02055           1.643      -0.1979           1.708        1.427       None  0.389665    507.9                   0.570
         HL63   done 0.02424           0.298      -0.4296           1.322       -0.474       None  0.444246    536.1                   0.296
        HL126   done 0.02569           0.550      -0.3636           1.253        0.146       None  0.431838   1033.5                   0.420
        HL252   done 0.02752           1.357      -0.1509           1.708        0.209       None  0.386094   1235.6                   0.519
        HL504   done 0.04527           1.495      -0.1271           1.755        0.861       None  0.332009   1022.7                   0.727
      HY3y126   done 0.02419           0.553      -0.3915           1.216        0.144       None  0.432172    229.9                   0.401
      HY3y252   done 0.02326           1.400      -0.2129           1.753        0.893       None  0.368538    256.0                   0.468
    CAL_bands   done 0.03629           1.054      -0.2028           1.631        0.370       None  0.397942    326.5                   0.486

