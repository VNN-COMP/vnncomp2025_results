#!/usr/bin/env python3
"""
Create clean TikZ plots using PGFPlots from accumulated data files
This generates TikZ code that doesn't require gnuplot-lua-tikz package
"""

import os
import glob
from pathlib import Path
import re

def get_tool_color_mapping():
    """Return a consistent color mapping for tools across all plots"""
    return {
        # Main VNN-COMP 2025 tools (based on actual accumulated files)
        'alpha_beta_crown': 'red',
        'cora': 'blue',
        'neuralsat': 'green',
        'nnenum': 'orange',
        'nnv': 'purple',
        'pyrat': 'brown',
        'sobolbox': 'cyan',
        'SCORING': 'black',
        # Additional colors for potential other tools
        'marabou': 'magenta',
        'ovalbab': 'pink',
        'verinet': 'gray',
        'mnbab': 'olive',
        'verapak': 'teal',
        'bab': 'lime',
        'prima': 'navy',
        'eran': 'maroon',
        'planet': 'darkgreen',
        'relu_analyzer': 'violet',
        'neurify': 'yellow',
        'venus': 'lightblue',
        'auto_lirpa': 'coral',
        'beta_crown': 'indigo',
        'oval': 'salmon',
        'rover': 'gold',
        'abcrown': 'red',  # alias for alpha_beta_crown
    }

def get_tool_line_style_mapping():
    """Return a consistent line style mapping for tools"""
    return {
        # Main VNN-COMP 2025 tools (based on actual accumulated files)
        'alpha_beta_crown': 'solid',
        'cora': 'dashed',
        'neuralsat': 'dotted',
        'nnenum': 'dashdotted',
        'nnv': 'densely dashed',
        'pyrat': 'loosely dashed',
        'sobolbox': 'loosely dotted',
        'SCORING': 'densely dotted',
        # Additional line styles for potential other tools
        'marabou': 'dashed',
        'ovalbab': 'dotted',
        'verinet': 'dashdotted',
        'mnbab': 'densely dashed',
        'verapak': 'loosely dashed',
        'bab': 'solid',
        'prima': 'dashed',
        'eran': 'dotted',
        'planet': 'dashdotted',
        'relu_analyzer': 'densely dashed',
        'neurify': 'loosely dashed',
        'venus': 'solid',
        'auto_lirpa': 'dashed',
        'beta_crown': 'dotted',
        'oval': 'dashdotted',
        'rover': 'densely dotted',
        'abcrown': 'solid',  # alias for alpha_beta_crown
    }

def normalize_tool_name(tool_name):
    """Normalize tool name for consistent mapping"""
    # Handling variations and aliases
    normalized = tool_name.lower().strip()
    normalized = normalized.replace('-', '_')
    normalized = normalized.replace('alpha-beta-crown', 'alpha_beta_crown')
    normalized = normalized.replace('abcrown', 'alpha_beta_crown')
    normalized = normalized.replace('beta-crown', 'beta_crown')
    return normalized

def create_pgfplot_tikz(category, data_files, output_file, category_info):
    """Creating a clean TikZ plot using PGFPlots with embedded data"""
    
    # Getting the consistent color and line style mappings
    color_mapping = get_tool_color_mapping()
    line_style_mapping = get_tool_line_style_mapping()
    
    # Fallback colors and line styles for unmapped tools
    fallback_colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    fallback_line_styles = ['solid', 'dashed', 'dotted', 'dashdotted', 'densely dashed', 'loosely dashed']
    
    # Read and process all data files
    all_tool_data = {}
    max_instances = 0
    max_time = 0
    min_time = float('inf')
    
    for tool, data_file in data_files.items():
        if os.path.exists(data_file) and os.path.getsize(data_file) > 0:  # Only process non-empty files
            tool_data = []
            with open(data_file, 'r') as f:
                for line in f:
                    if line.strip():
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            try:
                                time_val = float(parts[0])
                                instance_count = int(parts[1])
                                tool_data.append((instance_count, time_val))
                                max_instances = max(max_instances, instance_count)
                                max_time = max(max_time, time_val)
                                min_time = min(min_time, time_val)
                            except ValueError:
                                continue
            
            # Sort by instance count
            tool_data.sort()
            all_tool_data[tool] = tool_data
    
    # Plot limits
    ymin = max(0.01, min_time * 0.8) if min_time != float('inf') else 0.01
    ymax = max_time * 1.5 if max_time > 0 else 300
    xmax = max_instances * 1.1 if max_instances > 0 else 200
    
    # Timeout line
    timeout_y = 300 if max_time > 300 else 60
    timeout_label = "Five Minutes" if max_time > 300 else "One Minute"
    
    with open(output_file, 'w') as f:
        f.write(f"""% TikZ plot for {category}
% Generated automatically with embedded data
% Requires: \\usepackage{{pgfplots}} \\pgfplotsset{{compat=1.18}}

\\begin{{tikzpicture}}
\\begin{{semilogyaxis}}[
    xlabel={{Number of Instances Verified}},
    ylabel={{Time (sec)}},
    legend pos=outer north east,
    grid=major,
    width=10cm,
    height=6.5cm,
    ymin={ymin:.3f},
    ymax={ymax:.1f},
    xmin=0,
    xmax={xmax:.0f},
    line width=1.5pt,
    legend style={{font=\\footnotesize, cells={{anchor=west}}, draw=none}},
    xlabel style={{font=\\small}},
    ylabel style={{font=\\small}},
    title={{{category_info.get('title', category.replace('2025_', '').replace('_', ' ').title())}}},
    title style={{font=\\normalsize}}
]

""")
        
        # Plots for each tool with embedded data
        fallback_index = 0
        for tool in sorted(all_tool_data.keys()):
            if tool in all_tool_data and all_tool_data[tool]:
                # Normalize tool name and get consistent color/style
                normalized_tool = normalize_tool_name(tool)
                
                # Get color from mapping, fallback to list if not found
                if normalized_tool in color_mapping:
                    color = color_mapping[normalized_tool]
                else:
                    color = fallback_colors[fallback_index % len(fallback_colors)]
                
                # Get line style from mapping, fallback to list if not found
                if normalized_tool in line_style_mapping:
                    line_style = line_style_mapping[normalized_tool]
                else:
                    line_style = fallback_line_styles[fallback_index % len(fallback_line_styles)]
                
                # Clean tool name for legend
                clean_tool_name = tool.replace('_', '-')
                if clean_tool_name == 'alpha-beta-crown':
                    clean_tool_name = '$\\alpha$-$\\beta$-CROWN'
                elif clean_tool_name == 'alpha-beta-crown' or tool == 'alpha_beta_crown':
                    clean_tool_name = '$\\alpha$-$\\beta$-CROWN'
                
                # Write embedded coordinate data
                f.write(f"""\\addplot[color={color}, mark=none, {line_style}] coordinates {{
""")
                
                # Add coordinates (x=instances, y=time)
                for instance_count, time_val in all_tool_data[tool]:
                    f.write(f"    ({instance_count},{time_val:.6f})\n")
                
                f.write(f"""}};
\\addlegendentry{{{clean_tool_name}}}

""")
                fallback_index += 1
        
        # Add timeout line
        f.write(f"""% Timeout line
\\addplot[color=gray, dashed, mark=none, domain=0:{xmax:.0f}] {{{timeout_y}}};

% Add timeout label as text below the line
\\node[color=gray] at (axis cs:{xmax*0.15:.0f},{timeout_y * 0.9:.1f}) {{{timeout_label}}};

\\end{{semilogyaxis}}
\\end{{tikzpicture}}
""")

def get_category_info():
    """Return category information for titles and descriptions"""
    return {
        # Overall categories
        'all': {'title': 'All Instances'},
        'all_scored': {'title': 'All Scored Instances'},
        
        # VNN-COMP 2025 categories
        '2025_acasxu_2023': {'title': 'ACAS Xu 2023'},
        '2025_cctsdb_yolo_2023': {'title': 'CCTSDB YOLO 2023'},
        '2025_cersyve': {'title': 'Cersyve'},
        '2025_cgan_2023': {'title': 'CGAN 2023'},
        '2025_cifar100_2024': {'title': 'CIFAR-100 2024'},
        '2025_collins_aerospace_benchmark': {'title': 'Collins Aerospace Benchmark'},
        '2025_collins_rul_cnn_2022': {'title': 'Collins RUL CNN 2022'},
        '2025_cora_2024': {'title': 'CORA 2024'},
        '2025_dist_shift_2023': {'title': 'Distribution Shift 2023'},
        '2025_linearizenn_2024': {'title': 'LinearizeNN 2024'},
        '2025_lsnc_relu': {'title': 'LSNC ReLU'},
        '2025_malbeware': {'title': 'Malbeware'},
        '2025_metaroom_2023': {'title': 'MetaRoom 2023'},
        '2025_ml4acopf_2024': {'title': 'ML4ACOPF 2024'},
        '2025_nn4sys': {'title': 'NN4SYS'},
        '2025_relusplitter': {'title': 'ReLU Splitter'},
        '2025_safenlp_2024': {'title': 'SafeNLP 2024'},
        '2025_sat_relu': {'title': 'SAT ReLU'},
        '2025_soundnessbench': {'title': 'Soundness Bench'},
        '2025_tinyimagenet_2024': {'title': 'Tiny ImageNet 2024'},
        '2025_tllverifybench_2023': {'title': 'TLL Verify Bench 2023'},
        '2025_traffic_signs_recognition_2023': {'title': 'Traffic Signs Recognition 2023'},
        '2025_vggnet16_2022': {'title': 'VGGNet16 2022'},
        '2025_vit_2023': {'title': 'ViT 2023'},
        '2025_yolo_2023': {'title': 'YOLO 2023'},
        
        # Legacy 2024 categories (kept for compatibility)
        '2024_acasxu_2023': {'title': 'ACAS Xu 2023'},
        '2024_cctsdb_yolo_2023': {'title': 'CCTSDB YOLO 2023'},
        '2024_cgan_2023': {'title': 'CGAN 2023'},
        '2024_cifar100': {'title': 'CIFAR-100'},
        '2024_collins_aerospace_benchmark': {'title': 'Collins Aerospace Benchmark'},
        '2024_collins_rul_cnn_2023': {'title': 'Collins RUL CNN 2023'},
        '2024_cora': {'title': 'CORA'},
        '2024_dist_shift_2023': {'title': 'Distribution Shift 2023'},
        '2024_linearizenn': {'title': 'LinearizeNN'},
        '2024_lsnc': {'title': 'LSNC'},
        '2024_metaroom_2023': {'title': 'MetaRoom 2023'},
        '2024_ml4acopf_2023': {'title': 'ML4ACOPF 2023'},
        '2024_ml4acopf_2024': {'title': 'ML4ACOPF 2024'},
        '2024_nn4sys_2023': {'title': 'NN4SYS 2023'},
        '2024_safenlp': {'title': 'SafeNLP'},
        '2024_tinyimagenet': {'title': 'Tiny ImageNet'},
        '2024_tllverifybench_2023': {'title': 'TLLVerifyBench 2023'},
        '2024_traffic_signs_recognition_2023': {'title': 'Traffic Signs Recognition 2023'},
        '2024_vggnet16_2023': {'title': 'VGGNet16 2023'},
        '2024_vit_2023': {'title': 'ViT 2023'},
        '2024_yolo_2023': {'title': 'YOLO 2023'},
    }

def main():
    """Main function to create all TikZ plots"""
    plots_dir = "plots"
    output_dir = "latex/tikz"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all accumulated files
    pattern = f"{plots_dir}/accumulated-*.txt"
    files = glob.glob(pattern)
    
    if not files:
        print(f"No accumulated files found with pattern: {pattern}")
        print(f"Current directory: {os.getcwd()}")
        print(f"Files in plots directory: {os.listdir(plots_dir) if os.path.exists(plots_dir) else 'Directory does not exist'}")
        return
    
    print(f"Found {len(files)} accumulated data files")
    
    # Group files by category
    categories = {}
    for file in files:
        filename = Path(file).stem  # Remove .txt extension
        # Pattern: accumulated-{category}-{tool}
        match = re.match(r'accumulated-(.+?)-(.+)', filename)
        if match:
            category = match.group(1)
            tool = match.group(2)
            
            if category not in categories:
                categories[category] = {}
            categories[category][tool] = file
        else:
            print(f"Warning: Could not parse filename: {filename}")
    
    print(f"Found categories: {list(categories.keys())}")
    
    # Get category information
    category_info = get_category_info()
    
    # Create TikZ for each category
    created_count = 0
    for category, data_files in categories.items():
        if not data_files:
            print(f"Warning: No data files for category {category}")
            continue
            
        output_file = f"{output_dir}/{category}.tex"
        try:
            create_pgfplot_tikz(category, data_files, output_file, category_info)
            print(f"Created {output_file} with {len(data_files)} tools")
            created_count += 1
        except Exception as e:
            print(f"Error creating {output_file}: {e}")
    
    print(f"\nCreated {created_count} TikZ files in {output_dir}/")
    print(f"To use in LaTeX:")
    print(f"1. Add to preamble: \\usepackage{{pgfplots}} \\pgfplotsset{{compat=1.18}}")
    print(f"2. Include plots with: \\input{{{output_dir}/category.tex}}")

if __name__ == "__main__":
    main()
