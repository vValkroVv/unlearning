# DUET and RWKU Metric Tables

Numbers come from `DUET_SUMMARY.json` under `metrics-new/saves-clean/unlearn/.../evals/`.

Color key: <span style="color:#c62828"><strong>F</strong></span> = `forget_qa_rouge`, <span style="color:#1565c0"><strong>H</strong></span> = `holdout_qa_rouge`.

Selection note: `npo` uses the production default suffix `beta0p5_alpha1p0_gamma1p0`; `npo_sam` uses `beta0p1_alpha1p0_gamma1p0_rho0p01_adF`; DUET uses merged `city_forget_5`.

## DUET

### Llama-3.1-8B-Instruct

| Method | 1e-6 | 5e-6 | 1e-5 | 5e-5 | 1e-4 |
|---|---|---|---|---|---|
| GA | <span style="color:#c62828">F 0.9368</span><br><span style="color:#1565c0">H 0.9635</span> | <span style="color:#c62828">F 0.9453</span><br><span style="color:#1565c0">H 0.9567</span> | <span style="color:#c62828">F 0.8293</span><br><span style="color:#1565c0">H 0.9363</span> | <span style="color:#c62828">F 0.0000</span><br><span style="color:#1565c0">H 0.0000</span> | <span style="color:#c62828">F 0.0000</span><br><span style="color:#1565c0">H 0.0000</span> |
| NPO | <span style="color:#c62828">F 0.9363</span><br><span style="color:#1565c0">H 0.9608</span> | <span style="color:#c62828">F 0.9372</span><br><span style="color:#1565c0">H 0.9690</span> | <span style="color:#c62828">F 0.9136</span><br><span style="color:#1565c0">H 0.9650</span> | <span style="color:#c62828">F 0.8675</span><br><span style="color:#1565c0">H 0.9843</span> | <span style="color:#c62828">F 0.8339</span><br><span style="color:#1565c0">H 0.9953</span> |
| NPO-SAM | <span style="color:#c62828">F 0.9407</span><br><span style="color:#1565c0">H 0.9625</span> | <span style="color:#c62828">F 0.9409</span><br><span style="color:#1565c0">H 0.9620</span> | <span style="color:#c62828">F 0.9242</span><br><span style="color:#1565c0">H 0.9623</span> | <span style="color:#c62828">F 0.8404</span><br><span style="color:#1565c0">H 0.9775</span> | <span style="color:#c62828">F 0.6842</span><br><span style="color:#1565c0">H 0.9698</span> |
| LoKU | <span style="color:#c62828">F 0.8994</span><br><span style="color:#1565c0">H 0.9657</span> | <span style="color:#c62828">F 0.2319</span><br><span style="color:#1565c0">H 0.9393</span> | <span style="color:#c62828">F 0.2736</span><br><span style="color:#1565c0">H 0.9440</span> | <span style="color:#c62828">F 0.0621</span><br><span style="color:#1565c0">H 0.9440</span> | <span style="color:#c62828">F 0.0809</span><br><span style="color:#1565c0">H 0.9490</span> |

### Qwen2.5-7B-Instruct

| Method | 1e-6 | 5e-6 | 1e-5 | 5e-5 | 1e-4 |
|---|---|---|---|---|---|
| GA | <span style="color:#c62828">F 0.8846</span><br><span style="color:#1565c0">H 0.8348</span> | <span style="color:#c62828">F 0.8690</span><br><span style="color:#1565c0">H 0.8332</span> | <span style="color:#c62828">F 0.8124</span><br><span style="color:#1565c0">H 0.8202</span> | <span style="color:#c62828">F 0.0000</span><br><span style="color:#1565c0">H 0.0000</span> | <span style="color:#c62828">F 0.0000</span><br><span style="color:#1565c0">H 0.0000</span> |
| NPO | <span style="color:#c62828">F 0.8811</span><br><span style="color:#1565c0">H 0.8308</span> | <span style="color:#c62828">F 0.8729</span><br><span style="color:#1565c0">H 0.8365</span> | <span style="color:#c62828">F 0.8563</span><br><span style="color:#1565c0">H 0.8315</span> | <span style="color:#c62828">F 0.8003</span><br><span style="color:#1565c0">H 0.8780</span> | <span style="color:#c62828">F 0.6959</span><br><span style="color:#1565c0">H 0.9102</span> |
| NPO-SAM | <span style="color:#c62828">F 0.8852</span><br><span style="color:#1565c0">H 0.8305</span> | <span style="color:#c62828">F 0.8627</span><br><span style="color:#1565c0">H 0.8425</span> | <span style="color:#c62828">F 0.8387</span><br><span style="color:#1565c0">H 0.8385</span> | <span style="color:#c62828">F 0.5907</span><br><span style="color:#1565c0">H 0.8619</span> | <span style="color:#c62828">F 0.2107</span><br><span style="color:#1565c0">H 0.7688</span> |
| LoKU | <span style="color:#c62828">F 0.8674</span><br><span style="color:#1565c0">H 0.8507</span> | <span style="color:#c62828">F 0.7829</span><br><span style="color:#1565c0">H 0.8595</span> | <span style="color:#c62828">F 0.7163</span><br><span style="color:#1565c0">H 0.8820</span> | <span style="color:#c62828">F 0.1855</span><br><span style="color:#1565c0">H 0.8953</span> | <span style="color:#c62828">F 0.0738</span><br><span style="color:#1565c0">H 0.9122</span> |

### gemma-7b-it

| Method | 1e-6 | 5e-6 | 1e-5 | 5e-5 | 1e-4 |
|---|---|---|---|---|---|
| GA | <span style="color:#c62828">F 0.8795</span><br><span style="color:#1565c0">H 0.9043</span> | <span style="color:#c62828">F 0.7861</span><br><span style="color:#1565c0">H 0.8173</span> | <span style="color:#c62828">F 0.0624</span><br><span style="color:#1565c0">H 0.0095</span> | <span style="color:#c62828">F 0.0000</span><br><span style="color:#1565c0">H 0.0000</span> | <span style="color:#c62828">F 0.0000</span><br><span style="color:#1565c0">H 0.0000</span> |
| NPO | <span style="color:#c62828">F 0.8840</span><br><span style="color:#1565c0">H 0.9147</span> | <span style="color:#c62828">F 0.8747</span><br><span style="color:#1565c0">H 0.9292</span> | <span style="color:#c62828">F 0.8628</span><br><span style="color:#1565c0">H 0.9204</span> | <span style="color:#c62828">F 0.8071</span><br><span style="color:#1565c0">H 0.8520</span> | <span style="color:#c62828">F 0.7018</span><br><span style="color:#1565c0">H 0.9081</span> |
| NPO-SAM | <span style="color:#c62828">F 0.8769</span><br><span style="color:#1565c0">H 0.9087</span> | <span style="color:#c62828">F 0.8847</span><br><span style="color:#1565c0">H 0.9338</span> | <span style="color:#c62828">F 0.8599</span><br><span style="color:#1565c0">H 0.9313</span> | <span style="color:#c62828">F 0.6956</span><br><span style="color:#1565c0">H 0.8373</span> | <span style="color:#c62828">F 0.5389</span><br><span style="color:#1565c0">H 0.7770</span> |
| LoKU | <span style="color:#c62828">F 0.8217</span><br><span style="color:#1565c0">H 0.9343</span> | <span style="color:#c62828">F 0.1323</span><br><span style="color:#1565c0">H 0.8758</span> | <span style="color:#c62828">F 0.1031</span><br><span style="color:#1565c0">H 0.9165</span> | <span style="color:#c62828">F 0.1234</span><br><span style="color:#1565c0">H 0.9112</span> | <span style="color:#c62828">F 0.0805</span><br><span style="color:#1565c0">H 0.9017</span> |

## RWKU

### Llama-3.1-8B-Instruct

| Method | 1e-6 | 5e-6 | 1e-5 | 5e-5 | 1e-4 |
|---|---|---|---|---|---|
| GA | <span style="color:#c62828">F 0.8315</span><br><span style="color:#1565c0">H 0.8777</span> | <span style="color:#c62828">F 0.0080</span><br><span style="color:#1565c0">H 0.0070</span> | <span style="color:#c62828">F 0.0008</span><br><span style="color:#1565c0">H 0.0001</span> | <span style="color:#c62828">F 0.0030</span><br><span style="color:#1565c0">H 0.0009</span> | <span style="color:#c62828">F 0.0022</span><br><span style="color:#1565c0">H 0.0009</span> |
| NPO | <span style="color:#c62828">F 0.7580</span><br><span style="color:#1565c0">H 0.8314</span> | <span style="color:#c62828">F 0.7556</span><br><span style="color:#1565c0">H 0.8373</span> | <span style="color:#c62828">F 0.7512</span><br><span style="color:#1565c0">H 0.8487</span> | <span style="color:#c62828">F 0.6838</span><br><span style="color:#1565c0">H 0.8940</span> | <span style="color:#c62828">F 0.6318</span><br><span style="color:#1565c0">H 0.9086</span> |
| NPO-SAM | <span style="color:#c62828">F 0.7850</span><br><span style="color:#1565c0">H 0.8545</span> | <span style="color:#c62828">F 0.8331</span><br><span style="color:#1565c0">H 0.8915</span> | <span style="color:#c62828">F 0.8750</span><br><span style="color:#1565c0">H 0.8849</span> | <span style="color:#c62828">F 0.6815</span><br><span style="color:#1565c0">H 0.8709</span> | <span style="color:#c62828">F 0.5286</span><br><span style="color:#1565c0">H 0.8851</span> |
| LoKU | <span style="color:#c62828">F 0.6633</span><br><span style="color:#1565c0">H 0.8084</span> | <span style="color:#c62828">F 0.5064</span><br><span style="color:#1565c0">H 0.8548</span> | <span style="color:#c62828">F 0.1185</span><br><span style="color:#1565c0">H 0.7976</span> | <span style="color:#c62828">F 0.0689</span><br><span style="color:#1565c0">H 0.8995</span> | <span style="color:#c62828">F 0.0389</span><br><span style="color:#1565c0">H 0.8806</span> |

### Qwen2.5-7B-Instruct

| Method | 1e-6 | 5e-6 | 1e-5 | 5e-5 | 1e-4 |
|---|---|---|---|---|---|
| GA | <span style="color:#c62828">F 0.4270</span><br><span style="color:#1565c0">H 0.4795</span> | <span style="color:#c62828">F 0.4578</span><br><span style="color:#1565c0">H 0.4956</span> | <span style="color:#c62828">F 0.0119</span><br><span style="color:#1565c0">H 0.0061</span> | <span style="color:#c62828">F 0.0022</span><br><span style="color:#1565c0">H 0.0003</span> | <span style="color:#c62828">F 0.0008</span><br><span style="color:#1565c0">H 0.0001</span> |
| NPO | <span style="color:#c62828">F 0.4266</span><br><span style="color:#1565c0">H 0.4806</span> | <span style="color:#c62828">F 0.4162</span><br><span style="color:#1565c0">H 0.4823</span> | <span style="color:#c62828">F 0.4105</span><br><span style="color:#1565c0">H 0.4920</span> | <span style="color:#c62828">F 0.3739</span><br><span style="color:#1565c0">H 0.5639</span> | <span style="color:#c62828">F 0.3458</span><br><span style="color:#1565c0">H 0.6479</span> |
| NPO-SAM | <span style="color:#c62828">F 0.4266</span><br><span style="color:#1565c0">H 0.4778</span> | <span style="color:#c62828">F 0.4109</span><br><span style="color:#1565c0">H 0.4767</span> | <span style="color:#c62828">F 0.4006</span><br><span style="color:#1565c0">H 0.4781</span> | <span style="color:#c62828">F 0.3797</span><br><span style="color:#1565c0">H 0.5384</span> | <span style="color:#c62828">F 0.2977</span><br><span style="color:#1565c0">H 0.5962</span> |
| LoKU | <span style="color:#c62828">F 0.4611</span><br><span style="color:#1565c0">H 0.5337</span> | <span style="color:#c62828">F 0.4606</span><br><span style="color:#1565c0">H 0.5914</span> | <span style="color:#c62828">F 0.4274</span><br><span style="color:#1565c0">H 0.6426</span> | <span style="color:#c62828">F 0.3398</span><br><span style="color:#1565c0">H 0.7433</span> | <span style="color:#c62828">F 0.0302</span><br><span style="color:#1565c0">H 0.7528</span> |

### gemma-7b-it

| Method | 1e-6 | 5e-6 | 1e-5 | 5e-5 | 1e-4 |
|---|---|---|---|---|---|
| GA | <span style="color:#c62828">F 0.4805</span><br><span style="color:#1565c0">H 0.5334</span> | <span style="color:#c62828">F 0.0000</span><br><span style="color:#1565c0">H 0.0000</span> | <span style="color:#c62828">F 0.0000</span><br><span style="color:#1565c0">H 0.0000</span> | <span style="color:#c62828">F 0.0000</span><br><span style="color:#1565c0">H 0.0000</span> | <span style="color:#c62828">F 0.0000</span><br><span style="color:#1565c0">H 0.0000</span> |
| NPO | <span style="color:#c62828">F 0.4930</span><br><span style="color:#1565c0">H 0.5278</span> | <span style="color:#c62828">F 0.4825</span><br><span style="color:#1565c0">H 0.5339</span> | <span style="color:#c62828">F 0.4744</span><br><span style="color:#1565c0">H 0.5442</span> | <span style="color:#c62828">F 0.4159</span><br><span style="color:#1565c0">H 0.6159</span> | <span style="color:#c62828">F 0.2992</span><br><span style="color:#1565c0">H 0.6345</span> |
| NPO-SAM | <span style="color:#c62828">F 0.4973</span><br><span style="color:#1565c0">H 0.5378</span> | <span style="color:#c62828">F 0.4134</span><br><span style="color:#1565c0">H 0.5066</span> | <span style="color:#c62828">F 0.3406</span><br><span style="color:#1565c0">H 0.4810</span> | <span style="color:#c62828">F 0.3134</span><br><span style="color:#1565c0">H 0.5718</span> | <span style="color:#c62828">F 0.2712</span><br><span style="color:#1565c0">H 0.6441</span> |
| LoKU | <span style="color:#c62828">F 0.3149</span><br><span style="color:#1565c0">H 0.4535</span> | <span style="color:#c62828">F 0.3614</span><br><span style="color:#1565c0">H 0.7042</span> | <span style="color:#c62828">F 0.2127</span><br><span style="color:#1565c0">H 0.7381</span> | <span style="color:#c62828">F 0.0519</span><br><span style="color:#1565c0">H 0.8286</span> | <span style="color:#c62828">F 0.0200</span><br><span style="color:#1565c0">H 0.8371</span> |
