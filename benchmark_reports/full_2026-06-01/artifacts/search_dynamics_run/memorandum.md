# Evolution Memorandum

[OUTCOME HISTORY]
- ACCEPTED: step=1 score=0.970850 best=0.970850
- ACCEPTED: step=2 score=0.970850 best=0.970850
- ACCEPTED: step=3 score=0.970850 best=0.970850
- ACCEPTED: step=4 score=0.970850 best=0.970850
- ACCEPTED: step=5 score=0.970850 best=0.970850
- ACCEPTED: step=6 score=0.970850 best=0.970850
- ACCEPTED: step=7 score=0.970850 best=0.970850
- ACCEPTED: step=8 score=0.970850 best=0.970850

[STATE]
- Best AUC: 0.970850; config={'segment_stats': 'basic', 'trend_stats': 'linear', 'shape_stats': 'cusum', 'activation': 'identity', 'ridge_w': 'uniform', 'ridge_g': 'identity'}.
- Configuration search: 16/72 evaluated, capped=True.
- Representation: effective_rank=7.0041, mean_max_corr=0.9999, global_ridge_auc=0.984568.
- Cache: hits=133, entries=39, key=ancestor_conditioned_subpath.
- Dominant feature: output.activated.identity_std_log_ratio imp=0.0905, ind_auc=0.8008, resid=0.1273.
- Dominant subnode: segment_stats.basic imp=1.0000, features=21.
- Dominant alternative: segment_stats.basic age=9, participations=9.

[WHAT WORKS]
- ACCEPTED: step=5 score=0.970850 best=0.970850
- ACCEPTED: step=6 score=0.970850 best=0.970850
- ACCEPTED: step=7 score=0.970850 best=0.970850
- ACCEPTED: step=8 score=0.970850 best=0.970850
- Productive substructure: segment_stats.basic aggregates imp=1.0000.
- Productive substructure: trend_stats.linear aggregates imp=1.0000.
- Productive substructure: shape_stats.cusum aggregates imp=1.0000.

[WHAT FAILED]
- Risky feature: output.activated.identity_std_log_ratio redundancy=1.0000, stability=9.6829.
- Risky feature: output.raw_concat.std_log_ratio redundancy=1.0000, stability=9.6829.
- Risky feature: output.raw_concat.cusum_peak redundancy=1.0000, stability=5.6930.
- Risky feature: output.activated.identity_cusum_peak redundancy=1.0000, stability=5.6930.

[ERROR LOG]
- No runtime errors recorded.

TOON diagnostics:
```
context:
  scoring: configuration-based (best config AUC = evoforest score)
  best_config_auc: 0.970850
  global_ridge_auc: 0.984568
  config_auc_range: [0.9324417009602195, 0.9708504801097394]
  fold_auc_std: 0.014330
  effective_rank: 7.0041
  mean_max_corr: 0.9999
  shap_reconstruction_error: 0.00000000
  n_features_global: 336
  n_features_best_config: 21
  n_configs: 16
  n_configs_total: 72
scoring:
  auc: 0.970850
  config: {'segment_stats': 'basic', 'trend_stats': 'linear', 'shape_stats': 'cusum', 'activation': 'identity', 'ridge_w': 'uniform', 'ridge_g': 'identity'}
  search: {'evaluated': 16, 'total': 72, 'capped': True, 'auc_range': [0.9324417009602195, 0.9708504801097394], 'auc_mean': 0.9511531207133059, 'auc_std': 0.013901039130458181, 'best_config_auc': 0.9708504801097394, 'n_features_global': 336, 'n_features_best_config': 21, 'top_configs': [{'auc': 0.9708504801097394, 'n_features': 21, 'config': {'segment_stats': 'basic', 'trend_stats': 'linear', 'shape_stats': 'cusum', 'activation': 'identity', 'ridge_w': 'uniform', 'ridge_g': 'identity'}}, {'auc': 0.9687928669410151, 'n_features': 21, 'config': {'activation': 'clipped_linear', 'ridge_g': 'identity', 'ridge_w': 'uniform', 'segment_stats': 'basic', 'shape_stats': 'spectral', 'trend_stats': 'linear'}}, {'auc': 0.9681069958847737, 'n_features': 21, 'config': {'activation': 'identity', 'ridge_g': 'identity', 'ridge_w': 'uniform', 'segment_stats': 'robust_mutation_1', 'shape_stats': 'spectral', 'trend_stats': 'linear'}}, {'auc': 0.967764060356653, 'n_features': 21, 'config': {'activation': 'sigmoid_gate', 'ridge_g': 'huber', 'ridge_w': 'uniform', 'segment_stats': 'basic', 'shape_stats': 'spectral', 'trend_stats': 'linear'}}, {'auc': 0.9660493827160493, 'n_features': 21, 'config': {'activation': 'identity', 'ridge_g': 'huber', 'ridge_w': 'boundary_energy', 'segment_stats': 'robust', 'shape_stats': 'spectral', 'trend_stats': 'linear'}}, {'auc': 0.9639917695473251, 'n_features': 21, 'config': {'activation': 'clipped_linear', 'ridge_g': 'identity', 'ridge_w': 'uniform', 'segment_stats': 'basic', 'shape_stats': 'cusum', 'trend_stats': 'linear'}}, {'auc': 0.9506172839506173, 'n_features': 21, 'config': {'activation': 'identity', 'ridge_g': 'huber', 'ridge_w': 'boundary_energy', 'segment_stats': 'robust', 'shape_stats': 'cusum', 'trend_stats': 'linear'}}, {'auc': 0.9506172839506173, 'n_features': 21, 'config': {'activation': 'identity', 'ridge_g': 'huber', 'ridge_w': 'boundary_energy', 'segment_stats': 'robust_mutation_1', 'shape_stats': 'cusum', 'trend_stats': 'linear'}}], 'cache': {'hits': 133, 'misses': 39, 'entries': 39, 'shared_across_configurations': True, 'key': 'ancestor_conditioned_subpath'}}
  fitting: {'ridge_w': {'alternative': 'uniform', 'min': 1.0, 'max': 1.0, 'mean': 1.0, 'std': 0.0}, 'ridge_g': {'alternative': 'identity', 'rule': 'identity', 'description': 'No residual reweighting.', 'irls_steps_requested': 2, 'irls_steps_used_per_fold': [0, 0, 0], 'irls': [{'fold': 0, 'steps_used': 0, 'final_alpha': 10.0, 'iterations': []}, {'fold': 1, 'steps_used': 0, 'final_alpha': 10.0, 'iterations': []}, {'fold': 2, 'steps_used': 0, 'final_alpha': 31.622776601683793, 'iterations': []}]}}
  global_ridge: {'alpha': 10.0, 'auc': 0.9845679012345679, 'intercept': 0.5, 'prediction_std': 0.4162090647677037, 'residual_std': 0.25864987281894986, 'residual_reweighted': False, 'sample_weight_min': 1.0, 'sample_weight_max': 1.0, 'sample_weight_mean': 1.0, 'sample_weight_std': 0.0, 'irls_steps_used': 0, 'irls_iterations': [], 'contribution_reconstruction_error': 2.220446049250313e-16, 'mean_abs_contribution': 0.03535518317852853}
features[name,depth,imp,auc,sign,max_corr,n_hi_corr,most_corr,res,res2,rank,effect,stab,shap,cv_shap]:
  output.activated.identity_std_log_ratio,5,0.0905,0.8008,1,1.0000,1,output.raw_concat.std_log_ratio,0.1273,0.1149,0.5209,1.1960,9.6829,0.0948,0.0662
  output.raw_concat.std_log_ratio,4,0.0905,0.8008,1,1.0000,1,output.activated.identity_std_log_ratio,0.1273,0.1149,0.5209,1.1960,9.6829,0.0948,0.0662
  output.raw_concat.cusum_peak,4,0.0717,0.8333,1,1.0000,6,output.activated.identity_cusum_peak,0.0301,0.0151,0.5774,1.5718,5.6930,0.0857,0.0565
  output.activated.identity_cusum_peak,5,0.0717,0.8333,1,1.0000,6,output.raw_concat.cusum_peak,0.0301,0.0151,0.5774,1.5718,5.6930,0.0857,0.0565
  output.projection.global_projection,4,0.0693,0.8505,1,0.9988,6,output.raw_concat.cusum_peak,0.0289,0.0117,0.6071,1.6115,6.1060,0.0821,0.0544
  output.raw_concat.std_log_ratio_abs,4,0.0587,0.7239,1,1.0000,1,output.activated.identity_std_log_ratio_abs,0.1022,0.1149,0.3879,0.9724,2.9963,0.0530,0.0390
  output.activated.identity_std_log_ratio_abs,5,0.0587,0.7239,1,1.0000,1,output.raw_concat.std_log_ratio_abs,0.1022,0.1149,0.3879,0.9724,2.9963,0.0530,0.0390
  output.raw_concat.mean_delta_abs,4,0.0493,0.8090,1,1.0000,6,output.activated.identity_mean_delta_abs,0.0072,-0.0279,0.5352,1.3988,4.3908,0.0463,0.0375
  output.activated.identity_mean_delta_abs,5,0.0493,0.8090,1,1.0000,6,output.raw_concat.mean_delta_abs,0.0072,-0.0279,0.5352,1.3988,4.3908,0.0463,0.0375
  output.raw_concat.slope_delta_abs,4,0.0479,0.7027,1,1.0000,1,output.activated.identity_slope_delta_abs,0.0002,-0.0366,0.3511,0.8761,2.6883,0.0412,0.0315
  output.activated.identity_slope_delta_abs,5,0.0479,0.7027,1,1.0000,1,output.raw_concat.slope_delta_abs,0.0002,-0.0366,0.3511,0.8761,2.6883,0.0412,0.0315
  output.activated.identity_post_slope,5,0.0396,0.6135,-1,1.0000,3,output.raw_concat.post_slope,-0.0239,0.0285,0.1966,0.5499,4.4659,0.0379,0.0254
subnodes[name,n,imp,shap,max_auc,abs_shap,res_abs,red,stab]:
  segment_stats.basic,21,1.0000,1.0000,0.8505,0.0354,0.0380,0.9999,4.0884
  trend_stats.linear,21,1.0000,1.0000,0.8505,0.0354,0.0380,0.9999,4.0884
  shape_stats.cusum,21,1.0000,1.0000,0.8505,0.0354,0.0380,0.9999,4.0884
  output.activated,10,0.4654,0.4589,0.8333,0.0341,0.0384,1.0000,3.9875
  activation.identity,10,0.4654,0.4589,0.8333,0.0341,0.0384,1.0000,3.9875
  output.raw_concat,10,0.4654,0.4589,0.8333,0.0341,0.0384,1.0000,3.9875
  output.projection,1,0.0693,0.0821,0.8505,0.0610,0.0289,0.9988,6.1060
alternatives[name,age,evals,sel,n,imp,shap,max_auc,abs_shap,res_abs,red,stab,last_auc]:
  segment_stats.basic,9,9,9,21,1.0000,1.0000,0.8505,0.0354,0.0380,0.9999,4.0884,0.970850
  shape_stats.cusum,9,9,9,21,1.0000,1.0000,0.8505,0.0354,0.0380,0.9999,4.0884,0.970850
  trend_stats.linear,9,9,9,21,1.0000,1.0000,0.8505,0.0354,0.0380,0.9999,4.0884,0.970850
  activation.identity,9,9,9,10,0.4654,0.4589,0.8333,0.0341,0.0384,1.0000,3.9875,0.970850
  output.activated,9,9,0,10,0.4654,0.4589,0.8333,0.0341,0.0384,1.0000,3.9875,0.970850
  output.raw_concat,9,9,0,10,0.4654,0.4589,0.8333,0.0341,0.0384,1.0000,3.9875,0.970850
  output.projection,9,9,0,1,0.0693,0.0821,0.8505,0.0610,0.0289,0.9988,6.1060,0.970850
  activation.clipped_linear,9,0,0,0,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.970850
  activation.sigmoid_gate,9,0,0,0,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.970850
  ridge_g.huber,9,0,0,0,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.970850
  ridge_g.identity,9,9,9,0,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.970850
  ridge_w.boundary_energy,9,0,0,0,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.970850
```
