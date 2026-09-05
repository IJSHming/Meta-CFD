import numpy as np
from evaluator import AerodynamicEvaluator
from utils import get_kulfan_parameters_from_name

evaluator = AerodynamicEvaluator(reynolds=2.5e6, alpha=5.0, min_thickness=0.08, max_thickness=0.20)
init_theta = get_kulfan_parameters_from_name("naca2412")

# 1. 验证基准 NACA 2412
det = evaluator.evaluate_detailed(init_theta)
val = evaluator.evaluate(init_theta)
print("--- 1. NACA 2412 Baseline ---")
print(f"Raw L/D: {det['ld']:.2f}, Penalized: {det['ld_penalized']:.2f}, evaluate(): {val:.2f}")
print(f"Feasible: {det['is_feasible']}, Fatal: {det['is_fatal']}, Total Penalty: {det['total_penalty']:.2f}")
print(f"t_max: {det['t_max']*100:.2f}%, x_t_max: {det['x_t_max']:.2f}, Area: {det['area']:.4f}, LE_r: {det['le_radius']:.4f}, TE_deg: {det['te_angle']:.1f}")
assert det['is_feasible'] is True, "NACA 2412 must be feasible!"
assert det['total_penalty'] == 0.0, "NACA 2412 penalty must be 0!"
assert abs(val - det['ld']) < 1e-3, "Evaluate output should match raw ld when feasible!"
print(">>> Baseline Check PASSED!")

# 2. 验证过薄翼型 (例如 5% 厚度)
thin_theta = init_theta.copy()
thin_theta[0:8] *= 0.42  # 压扁上表面
thin_theta[8:16] *= 0.42  # 压扁下表面
det_thin = evaluator.evaluate_detailed(thin_theta)
val_thin = evaluator.evaluate(thin_theta)
print("\n--- 2. Ultra-Thin Airfoil (5% thickness) ---")
print(f"Raw L/D: {det_thin['ld']:.2f}, Penalized L/D: {val_thin:.2f}, t_max: {det_thin['t_max']*100:.2f}%")
print(f"Feasible: {det_thin['is_feasible']}, Violations: {list(det_thin['violations'].keys())}")
print(f"Total Penalty: {det_thin['total_penalty']:.2f}")
assert det_thin['is_feasible'] is False, "Thin airfoil must not be feasible!"
assert "t_max_min" in det_thin['violations'], "Should detect t_max_min violation!"
assert val_thin < 0.0, f"Penalized fitness ({val_thin:.2f}) must be negative, completely eliminating thin-airfoil exploitation!"
print(">>> Thin Airfoil Penalty PASSED!")

# 3. 验证致命交叉翼型
bad_theta = init_theta.copy()
bad_theta[0:8] -= 0.3   # 上表面下陷
bad_theta[8:16] += 0.3  # 下表面隆起造成交叉
det_bad = evaluator.evaluate_detailed(bad_theta)
val_bad = evaluator.evaluate(bad_theta)
print("\n--- 3. Fatal Cross-Surface Airfoil ---")
print(f"Fatal: {det_bad['is_fatal']}, evaluate(): {val_bad:.2f}")
assert det_bad['is_fatal'] is True, "Should detect fatal defect!"
assert val_bad < -100.0, "Fatal airfoil must receive heavy negative score without NeuralFoil calculation!"
print(">>> Fatal Crossing Check PASSED!")

# 4. 验证后缘鱼尾/鼓包检测 (Fish-tailing)
fish_theta = init_theta.copy()
fish_theta[3] -= 0.2
fish_theta[4] -= 0.3
fish_theta[5] += 0.2
fish_theta[11] += 0.2
fish_theta[12] += 0.3
fish_theta[13] -= 0.2
det_fish = evaluator.evaluate_detailed(fish_theta)
print("\n--- 4. Fish-tailing Airfoil ---")
print(f"Feasible: {det_fish['is_feasible']}, Violations: {list(det_fish['violations'].keys())}")
assert "aft_non_mono" in det_fish['violations'], "Should catch aft fish-tailing!"
print(">>> Fish-tailing Check PASSED!")

# 5. 验证过厚翼型 (例如 24% 厚度)
thick_theta = init_theta.copy()
thick_theta[0:8] *= 2.0
thick_theta[8:16] *= 2.0
det_thick = evaluator.evaluate_detailed(thick_theta)
val_thick = evaluator.evaluate(thick_theta)
print("\n--- 5. Ultra-Thick Airfoil (24% thickness) ---")
print(f"Raw L/D: {det_thick['ld']:.2f}, Penalized L/D: {val_thick:.2f}, t_max: {det_thick['t_max']*100:.2f}%")
print(f"Feasible: {det_thick['is_feasible']}, Violations: {list(det_thick['violations'].keys())}")
assert det_thick['is_feasible'] is False
assert "t_max_max" in det_thick['violations']
print(">>> Thick Airfoil Penalty PASSED!")

print("\n==================================================")
print("ALL 5 INDUSTRIAL CONSTRAINT TESTS PASSED PERFECTLY!")
print("==================================================")
