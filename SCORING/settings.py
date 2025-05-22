'''
vnn comp global settings
'''

import os

from pathlib import Path

class GnuplotSettings:
    """settings for gnuplot"""

    def __init__(self, prefix, title=None):
        self.prefix = prefix

        if title is not None:
            self.title = title
        else:
            self.title = prefix

class Settings:
    '''static container for settings'''

    CSV_GLOB = "../*/results.csv"
    TOOL_LIST_GLOB_INDEX = 1

    SCORING_MIN_TIME = 1.0
    PLOT_MIN_TIME = 0 #0.01

    # latex or plotting may break if there are no unscored benchmarks
    REGULAR_BENCHMARKS = [
       "2024_safenlp",
       "2024_nn4sys_2023",
       "2024_cora",
       "2024_linearizenn",
       "2024_dist_shift_2023",
       "2024_cifar100",
       "2024_tinyimagenet",
       "2024_acasxu_2023",
       "2024_cgan_2023",
       "2024_collins_rul_cnn_2023",
       "2024_metaroom_2023",
       "2024_tllverifybench_2023",
    ]
    EXTENDED_BENCHMARKS = [
       "2024_ml4acopf_2023",
       "2024_ml4acopf_2024",
       "2024_collins_aerospace_benchmark",
       "2024_lsnc",
       "2024_yolo_2023",
       "2024_cctsdb_yolo_2023",
       "2024_traffic_signs_recognition_2023",
       "2024_vggnet16_2023",
       "2024_vit_2023",
    ]
    # Set UNSCORED_CATEGORIES to the set of benchmarks that should *not* be scored
    # So EXTENDED_BENCHMARKS for the regular track, and REGULAR_BENCHMARKS for the extended track
    UNSCORED_CATEGORIES = EXTENDED_BENCHMARKS
    ALWAYS_CHECK_COUNTEREXAMPLES = True

    SKIP_CE_FILES = True

    COUNTEREXAMPLE_ATOL = 1e-4 # used to check if CE satisfies spec (is insice input box and violates output property)
    COUNTEREXAMPLE_RTOL = 1e-3 # used to check if CE matches execution
    IGNORE_CE_Y = False  # Discard the Y provided in the CE files. Instead compute it using onnx

    PENALTY_INCORRECT = -150

    TOOL_NAME_SUBS_LATEX = [
            ('alpha_beta_crown', '$\\alpha$-$\\beta$-CROWN'),
            ('mn_bab', 'MN BaB'),
            ('fastbatllnn', 'FastBATLLNN'),
            ('neuralsat', 'NeuralSAT'),
            ('nnenum', 'nnenum'),
            ('nnv', 'NNV'),
            ('pyrat', 'PyRAT'),
            ('never2', 'NeVer2'),
            ('cora', 'CORA'),
            ]

    TOOL_NAME_SUBS_LONGTABLE = [
            ('alpha_beta_crown', '$\\alpha$-$\\beta$-C'),
            ('fastbatllnn', 'FastBaT'),
            ('marabou', 'Marab'),
            ('neuralsat', 'NSAT'),
            ('nnenum', 'NNen'),
            ('nnv', 'NNV'),
            ('pyrat', 'PyRAT'),
            ('never2', 'NeVer2'),
            ('cora', 'CORA'),
            ]

    TOOL_NAME_SUBS_GNUPLOT = [
            ('alpha_beta_crown', 'AB-CROWN'),
            ('fastbatllnn', 'FastBATLLNN'),
            ('neuralsat', 'NeuralSAT'),
            ('nnenum', 'nnenum'),
            ('nnv', 'NNV'),
            ('pyrat', 'PyRAT'),
        ]

    CAT_NAME_SUBS_LATEX = [
        ('carvana_unet_2022', 'Carvana 2022'),
        ('cifar100_tinyimagenet_resnet', 'Cifar100 Tiny'),
        ('reach_prob_density', 'Reach Prob Den~')
        ]

    SKIP_TOOLS = [] #['marabou', 'verapak', 'cgdtest']

    SKIP_BENCHMARK_TUPLES = [] #[('marabou', 'sri_resnet_a'), ('marabou', 'sri_resnet_b')]

    PLOTS_DIR = "./plots"

    CSV_SUBSTITUTIONS = [('unsat', 'holds'),
                         ('sat', 'violated'),
                         ('no_result_in_file', 'unknown'),
                         ('prepare_instance_error_', 'unknown'),
                         ('run_instance_timeout', 'timeout'),
                         ('prepare_instance_timeout', 'timeout'),
                         ('error_exit_code_', 'error'),
                         ('error_nonmaximal', 'unknown'),
                         ]

    # list of triples to override result if manually determined incorrect:
    # (cat_prefix, index, desired_result)
    OVERRIDE_RESULTS = [] #[('collins', 20, 'sat*')]

    # latex output files
    TOTAL_SCORE_LATEX = "latex/total.tex"
    SCORED_LATEX = "latex/scored.tex"
    UNSCORED_LATEX = "latex/unscored.tex"
    STATS_LATEX = "latex/stats.tex"
    LONGTABLE_LATEX = "latex/longtable.tex"

    # gnuplot information
    PLOT_FOLDER = "cactus" # folder containing the .pdfs
    
    gnuplot_data = (
        GnuplotSettings('all', 'All Instances'),
        GnuplotSettings('all_scored', 'All Scored Instances'),
        GnuplotSettings("2024_acasxu_2023", "acasxu 2023"),
        GnuplotSettings("2024_cctsdb_yolo_2023", "cctsdb yolo 2023"),
        GnuplotSettings("2024_cgan_2023", "cgan 2023"),
        GnuplotSettings("2024_cifar100", "cifar100"),

        GnuplotSettings("2024_collins_aerospace_benchmark", "collins aerospace benchmark"),
        GnuplotSettings("2024_collins_rul_cnn_2023", "collins rul cnn 2023"),
        GnuplotSettings("2024_cora", "cora"),
        GnuplotSettings("2024_dist_shift_2023", "dist shift 2023"),
        GnuplotSettings("2024_linearizenn", "linearizenn"),
        GnuplotSettings("2024_lsnc", "lsnc"),
        GnuplotSettings("2024_metaroom_2023", "metaroom 2023"),
        GnuplotSettings("2024_ml4acopf_2023", "ml4acopf 2023"),
        GnuplotSettings("2024_ml4acopf_2024", "ml4acopf 2024"),
        GnuplotSettings("2024_nn4sys_2023", "nn4sys 2023"),
        GnuplotSettings("2024_safenlp", "safenlp"),
        # GnuplotSetting  "2024_test", s("test"),
        GnuplotSettings("2024_tinyimagenet", "tinyimagenet"),
        GnuplotSettings("2024_tllverifybench_2023", "tllverifybench 2023"),
        GnuplotSettings("2024_traffic_signs_recognition_2023", "traffic signs recognition 2023"),
        GnuplotSettings("2024_vggnet16_2023", "vggnet16 2023"),
        GnuplotSettings("2024_vit_2023", "vit 2023"),
        GnuplotSettings("2024_yolo_2023", "yolo 2023"),

        #2022_carvana_unet_2022 2022_cifar100_tinyimagenet_resnet 2022_cifar2020 2022_cifar_biasfield 2022_mnist_fc 2022_nn4sys 2022_oval21 2022_reach_prob_density 2022_rl_benchmarks 2022_sri_resnet_a 2022_sri_resnet_b 2022_tllverifybench 2022_vggnet16_2022 2023_acasxu 2023_cctsdb_yolo 2023_cgan 2023_collins_rul_cnn 2023_collins_yolo_robustness 2023_dist_shift 2023_metaroom 2023_ml4acopf 2023_nn4sys 2023_test 2023_tllverifybench 2023_traffic_signs_recognition 2023_vggnet16 2023_vit 2023_yolo
#        GnuplotSettings('2022_carvana_unet_2022', 'Carvana Unet 2022'),
#        GnuplotSettings('2022_cifar100_tinyimagenet_resnet', 'CIFAR100 Tiny ImageNet ResNet'),
#        GnuplotSettings('2022_cifar2020', 'CIFAR2020'),
#        GnuplotSettings('2022_cifar_biasfield', 'CIFAR Biasfield'),
#        GnuplotSettings('2022_mnist_fc', 'MNIST FC'),
#        GnuplotSettings('2022_nn4sys', 'NN4SYS'),
#        GnuplotSettings('2022_oval21', 'OVAL 21'),
#        GnuplotSettings('2022_reach_prob_density', 'Reachability Probability Density'),
#        GnuplotSettings('2022_rl_benchmarks', 'Reinforcement Learning Benchmarks'),
#        GnuplotSettings('2022_sri_resnet_a', 'SRI Resnet A'),
#        GnuplotSettings('2022_sri_resnet_b', 'SRI Resnet B'),
#        GnuplotSettings('2022_tllverifybench', 'TLL Verify Bench (2022)'),
#        GnuplotSettings('2022_vggnet16_2022', 'VGGNet16 2022'),
#        GnuplotSettings('2023_acasxu', 'ACAS Xu'),
#        GnuplotSettings('2023_cctsdb_yolo', 'CCTSDb YOLO'),
#        GnuplotSettings('2023_cgan', 'CGAN'),
#        GnuplotSettings('2023_collins_rul_cnn', 'Collins Rul CNN'),
#        GnuplotSettings('2023_collins_yolo_robustness', 'Collins YOLO Robustness'),
#        GnuplotSettings('2023_dist_shift', 'Dist Shift'),
#        GnuplotSettings('2023_metaroom', 'Metaroom'),
#        GnuplotSettings('2023_ml4acopf', 'ML4ACOPF'),
#        GnuplotSettings('2023_nn4sys', 'NN4SYS'),
#        #GnuplotSettings('2023_test', 'Test'),
#        GnuplotSettings('2023_tllverifybench', 'TLL Verify Bench (2023)'),
#        GnuplotSettings('2023_traffic_signs_recognition', 'Traffic Signs Recognition'),
#        GnuplotSettings('2023_vggnet16', 'VGGNet16'),
#        GnuplotSettings('2023_vit', 'ViT'),
#        GnuplotSettings('2023_yolo', 'YOLO'),
        )
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    BENCHMARK_REPOS = {
        "2024": f"{base_dir}/vnncomp2024_benchmarks",
    }
    
for r in Settings.BENCHMARK_REPOS.values():
    assert Path(r).is_dir(), f"directory in Settings.BENCHMARK_REPOS ('{r}') " + \
        "doesn't exist. Please clone appropriate benchmark repo and edit " + \
        "path in Settings.BENCHMARK_REPOS in settings.py"
