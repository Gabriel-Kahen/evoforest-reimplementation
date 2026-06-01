# Evolution Memorandum

[OUTCOME HISTORY]
- REJECTED: step=1 score=0.969907 best=0.976080

[STATE]
- Best AUC: 0.976080; config={'activation': 'sigmoid_gate', 'ridge_g': 'identity', 'ridge_w': 'boundary_energy', 'segment_stats': 'basic', 'shape_stats': 'spectral', 'trend_stats': 'linear'}.
- Configuration search: 4/48 evaluated, capped=True.
- Representation: effective_rank=8.3175, mean_max_corr=0.9843, global_ridge_auc=0.996914.
- Cache: hits=29, entries=19, key=ancestor_conditioned_subpath.
- Dominant feature: output.activated.sigmoid_gate_pre_slope imp=0.1361, ind_auc=0.5309, resid=0.0190.
- Dominant subnode: segment_stats.basic imp=1.0000, features=21.
- Dominant alternative: segment_stats.basic age=1, participations=1.

[WHAT WORKS]
- Productive substructure: segment_stats.basic aggregates imp=1.0000.
- Productive substructure: trend_stats.linear aggregates imp=1.0000.
- Productive substructure: shape_stats.spectral aggregates imp=1.0000.

[WHAT FAILED]
- REJECTED: step=1 score=0.969907 best=0.976080
- Risky feature: output.activated.sigmoid_gate_pre_slope redundancy=1.0000, stability=0.7137.
- Risky feature: output.raw_concat.pre_slope redundancy=1.0000, stability=0.7461.
- Risky feature: output.raw_concat.std_log_ratio redundancy=0.9997, stability=0.9792.
- Risky feature: output.activated.sigmoid_gate_std_log_ratio redundancy=0.9997, stability=0.8921.

[ERROR LOG]
- No runtime errors recorded.

TOON diagnostics:
```
context:
  scoring: configuration-based (best config AUC = evoforest score)
  best_config_auc: 0.976080
  global_ridge_auc: 0.996914
  config_auc_range: [0.9591049382716049, 0.9760802469135802]
  fold_auc_std: 0.009821
  effective_rank: 8.3175
  mean_max_corr: 0.9843
  shap_reconstruction_error: 0.00000000
  n_features_global: 84
  n_features_best_config: 21
  n_configs: 4
  n_configs_total: 48
scoring:
  auc: 0.976080
  config: {'activation': 'sigmoid_gate', 'ridge_g': 'identity', 'ridge_w': 'boundary_energy', 'segment_stats': 'basic', 'shape_stats': 'spectral', 'trend_stats': 'linear'}
  search: {'evaluated': 4, 'total': 48, 'capped': True, 'auc_range': [0.9591049382716049, 0.9760802469135802], 'auc_mean': 0.9670138888888888, 'auc_std': 0.006510238051962269, 'best_config_auc': 0.9760802469135802, 'n_features_global': 84, 'n_features_best_config': 21, 'top_configs': [{'auc': 0.9760802469135802, 'n_features': 21, 'config': {'activation': 'sigmoid_gate', 'ridge_g': 'identity', 'ridge_w': 'boundary_energy', 'segment_stats': 'basic', 'shape_stats': 'spectral', 'trend_stats': 'linear'}}, {'auc': 0.9699074074074074, 'n_features': 21, 'config': {'segment_stats': 'basic', 'trend_stats': 'linear', 'shape_stats': 'cusum', 'activation': 'identity', 'ridge_w': 'uniform', 'ridge_g': 'identity'}}, {'auc': 0.9629629629629629, 'n_features': 21, 'config': {'activation': 'sigmoid_gate', 'ridge_g': 'huber', 'ridge_w': 'uniform', 'segment_stats': 'basic', 'shape_stats': 'cusum', 'trend_stats': 'linear'}}, {'auc': 0.9591049382716049, 'n_features': 21, 'config': {'activation': 'clipped_linear', 'ridge_g': 'huber', 'ridge_w': 'uniform', 'segment_stats': 'basic', 'shape_stats': 'cusum', 'trend_stats': 'linear'}}], 'cache': {'hits': 29, 'misses': 19, 'entries': 19, 'shared_across_configurations': True, 'key': 'ancestor_conditioned_subpath'}}
  fitting: {'ridge_w': {'alternative': 'boundary_energy', 'min': 0.4470845153157063, 'max': 1.3389269727836857, 'mean': 1.0, 'std': 0.2835252944557492}, 'ridge_g': {'alternative': 'identity', 'rule': 'identity', 'description': 'No residual reweighting.', 'irls_steps_requested': 2, 'irls_steps_used_per_fold': [0, 0, 0], 'irls': [{'fold': 0, 'steps_used': 0, 'final_alpha': 0.0031622776601683794, 'iterations': []}, {'fold': 1, 'steps_used': 0, 'final_alpha': 0.0001, 'iterations': []}, {'fold': 2, 'steps_used': 0, 'final_alpha': 0.01, 'iterations': []}]}}
  global_ridge: {'alpha': 0.01, 'auc': 0.9969135802469136, 'intercept': 0.4850665473924701, 'prediction_std': 0.44254361386554597, 'residual_std': 0.21982793476846446, 'residual_reweighted': False, 'sample_weight_min': 0.4470845153157063, 'sample_weight_max': 1.3389269727836857, 'sample_weight_mean': 1.0, 'sample_weight_std': 0.2835252944557492, 'irls_steps_used': 0, 'irls_iterations': [], 'contribution_reconstruction_error': 1.1102230246251565e-15, 'mean_abs_contribution': 0.6500861913356522}
features[name,depth,imp,auc,sign,max_corr,n_hi_corr,most_corr,res,res2,rank,effect,stab,shap,cv_shap]:
  output.activated.sigmoid_gate_pre_slope,5,0.1361,0.5309,-1,1.0000,1,output.raw_concat.pre_slope,0.0190,0.0289,-0.0535,-0.1483,0.7137,0.0144,5.9854
  output.raw_concat.pre_slope,4,0.1270,0.5309,1,1.0000,1,output.activated.sigmoid_gate_pre_slope,0.0191,0.2722,-0.0535,-0.1480,0.7461,0.0131,5.5378
  output.raw_concat.std_log_ratio,4,0.0785,0.7215,1,0.9997,1,output.activated.sigmoid_gate_std_log_ratio,-0.0927,-0.0452,0.3836,0.9083,0.9792,0.0609,3.6581
  output.activated.sigmoid_gate_std_log_ratio,5,0.0691,0.7215,-1,0.9997,1,output.raw_concat.std_log_ratio,-0.0934,-0.0892,0.3836,0.9037,0.8921,0.0420,3.3082
  output.raw_concat.slope_delta,4,0.0640,0.5224,-1,0.9999,1,output.activated.sigmoid_gate_slope_delta,0.1120,-0.0504,0.0388,0.2621,0.9152,0.0408,2.7021
  output.activated.sigmoid_gate_slope_delta,5,0.0471,0.5224,1,0.9999,1,output.raw_concat.slope_delta,0.1128,0.1036,0.0388,0.2591,1.6203,0.0462,1.8669
  output.activated.sigmoid_gate_std_log_ratio_abs,5,0.0456,0.7924,1,0.9996,1,output.raw_concat.std_log_ratio_abs,-0.0349,-0.0365,0.5066,1.1192,0.5214,0.0357,1.6432
  output.activated.sigmoid_gate_mean_delta_abs,5,0.0447,0.7978,1,0.9998,3,output.raw_concat.mean_delta_abs,-0.0274,-0.0287,0.5159,1.4372,1.5289,0.1127,1.8412
  output.raw_concat.std_log_ratio_abs,4,0.0438,0.7924,-1,0.9996,1,output.activated.sigmoid_gate_std_log_ratio_abs,-0.0360,-0.0452,0.5066,1.1118,0.4428,0.0480,1.5845
  output.activated.sigmoid_gate_mean_delta,5,0.0402,0.8079,1,0.9998,3,output.raw_concat.mean_delta,-0.0053,-0.0095,0.5333,1.3879,1.4792,0.0968,1.5726
  output.activated.sigmoid_gate_high_low_delta_abs,5,0.0393,0.7068,1,0.9992,1,output.raw_concat.high_low_delta_abs,0.0439,0.0401,0.3582,0.8031,3.4177,0.0767,1.7145
  output.raw_concat.mean_delta_abs,4,0.0369,0.7978,-1,0.9998,3,output.activated.sigmoid_gate_mean_delta_abs,-0.0290,-0.0385,0.5159,1.4265,1.3706,0.0928,1.5152
subnodes[name,n,imp,shap,max_auc,abs_shap,res_abs,red,stab]:
  segment_stats.basic,21,1.0000,1.0000,0.8079,0.6501,0.0606,0.9843,1.1938
  trend_stats.linear,21,1.0000,1.0000,0.8079,0.6501,0.0606,0.9843,1.1938
  shape_stats.spectral,21,1.0000,1.0000,0.8079,0.6501,0.0606,0.9843,1.1938
  output.activated,10,0.5099,0.5029,0.8079,0.6865,0.0592,0.9998,1.2301
  activation.sigmoid_gate,10,0.5099,0.5029,0.8079,0.6865,0.0592,0.9998,1.2301
  output.raw_concat,10,0.4716,0.4565,0.8079,0.6231,0.0593,0.9998,1.1093
  output.projection,1,0.0185,0.0407,0.6844,0.5550,0.0871,0.6757,1.6758
alternatives[name,age,evals,sel,n,imp,shap,max_auc,abs_shap,res_abs,red,stab,last_auc]:
  segment_stats.basic,1,1,1,21,1.0000,1.0000,0.8079,0.6501,0.0606,0.9843,1.1938,0.976080
  shape_stats.spectral,1,1,1,21,1.0000,1.0000,0.8079,0.6501,0.0606,0.9843,1.1938,0.976080
  trend_stats.linear,1,1,1,21,1.0000,1.0000,0.8079,0.6501,0.0606,0.9843,1.1938,0.976080
  activation.sigmoid_gate,1,1,1,10,0.5099,0.5029,0.8079,0.6865,0.0592,0.9998,1.2301,0.976080
  output.activated,1,1,0,10,0.5099,0.5029,0.8079,0.6865,0.0592,0.9998,1.2301,0.976080
  output.raw_concat,1,1,0,10,0.4716,0.4565,0.8079,0.6231,0.0593,0.9998,1.1093,0.976080
  output.projection,1,1,0,1,0.0185,0.0407,0.6844,0.5550,0.0871,0.6757,1.6758,0.976080
  activation.clipped_linear,1,0,0,0,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.976080
  activation.identity,1,0,0,0,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.976080
  ridge_g.huber,1,0,0,0,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.976080
  ridge_g.identity,1,1,1,0,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.976080
  ridge_w.boundary_energy,1,1,1,0,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.976080
```
