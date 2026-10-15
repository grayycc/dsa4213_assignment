Paired bootstrap over test abstracts (same resamples for both sides).

| A | B | macro-F1 A-B | 95% CI | p (two-sided) |
|---|---|---|---|---|
| transformer:mean | lstm:mean | +0.033 | [+0.026, +0.041] | 0.000 |
| transformer:mean | rnn:mean | +0.034 | [+0.027, +0.042] | 0.000 |
| rnn:mean | lstm:mean | -0.001 | [-0.009, +0.006] | 0.716 |
| transformer:last | lstm:last | +0.079 | [+0.070, +0.087] | 0.000 |
| transformer:last | rnn:last | +0.107 | [+0.098, +0.116] | 0.000 |
| rnn:last | lstm:last | -0.028 | [-0.036, -0.019] | 0.000 |
| rnn:mean | rnn:last | +0.089 | [+0.080, +0.098] | 0.000 |
| lstm:mean | lstm:last | +0.062 | [+0.054, +0.071] | 0.000 |
| transformer:mean | transformer:last | +0.016 | [+0.010, +0.023] | 0.000 |
| lstm_w2v:mean | lstm_w2v:last | +0.070 | [+0.062, +0.078] | 0.000 |
| lstm_glove:mean | lstm_glove:last | +0.070 | [+0.060, +0.078] | 0.000 |
| lstm_randfrozen:mean | lstm_randfrozen:last | +0.064 | [+0.056, +0.072] | 0.000 |
| lstm_glove_ft:mean | lstm_glove_ft:last | +0.072 | [+0.063, +0.079] | 0.000 |
| rnn:mean | rnn:emb | +0.094 | [+0.085, +0.102] | 0.000 |
| lstm:mean | lstm:emb | +0.101 | [+0.093, +0.110] | 0.000 |
| transformer:mean | transformer:emb | +0.122 | [+0.113, +0.131] | 0.000 |
| lstm_w2v:mean | lstm_w2v:emb | +0.037 | [+0.029, +0.044] | 0.000 |
| lstm_glove:mean | lstm_glove:emb | +0.052 | [+0.044, +0.061] | 0.000 |
| lstm_randfrozen:mean | lstm_randfrozen:emb | +0.166 | [+0.157, +0.175] | 0.000 |
| lstm_glove_ft:mean | lstm_glove_ft:emb | +0.046 | [+0.037, +0.055] | 0.000 |
| lstm_w2v:mean | lstm:mean | +0.040 | [+0.032, +0.048] | 0.000 |
| lstm_glove:mean | lstm:mean | -0.003 | [-0.011, +0.006] | 0.452 |
| lstm_randfrozen:mean | lstm:mean | -0.007 | [-0.015, +0.000] | 0.056 |
| lstm_glove_ft:mean | lstm:mean | +0.009 | [+0.002, +0.017] | 0.024 |
| lstm_glove:mean | lstm_w2v:mean | -0.043 | [-0.051, -0.035] | 0.000 |
| lstm_w2v:mean | transformer:mean | +0.007 | [-0.000, +0.014] | 0.058 |
| rnn:mean | tfidf | -0.059 | [-0.067, -0.051] | 0.000 |
| lstm:mean | tfidf | -0.058 | [-0.066, -0.050] | 0.000 |
| transformer:mean | tfidf | -0.025 | [-0.032, -0.017] | 0.000 |
