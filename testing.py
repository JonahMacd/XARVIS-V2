import matplotlib.pyplot as plt
from Jonah_Code_File.main import run
from Jonah_Code_File.spectrumanalyser import SpectrumAnalyzer

# 1. Run the star pipeline (asteroid=None) with a small subset for a fast test
results = run(asteroid=None, max_images=50, show=True)

# 2. Initialize the analyzer
analyzer = SpectrumAnalyzer(
    results['frames'],
    brightness=results.get('brightness'),
    spectrum=results.get('spectrum'),
)

# 3. Test 1: Compare two arbitrary captures (T1 vs T10)
report_diff = analyzer.compare(0, 9)
print("\n--- T1 vs T10 ---")
print(report_diff['comparison'])
analyzer.plot_diagnostic_comparison(results['zooms'], idx_a=0, idx_b=9, tracked=results['tracked'])

# 4. Test 2: Auto-pair T1 with its closest-wavelength capture
partner = analyzer.find_comparison_partner(0)
if partner is not None:
    report_same = analyzer.compare(0, partner)
    print(f"\n--- T1 vs T{partner + 1} (Closest Wavelength) ---")
    print(report_same['comparison'])
    analyzer.plot_diagnostic_comparison(results['zooms'], idx_a=0, idx_b=partner, tracked=results['tracked'])

plt.show()