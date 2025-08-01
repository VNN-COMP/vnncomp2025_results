"""
Process vnncomp results

Stanley Bak
"""

from typing import Dict, List, Tuple, Union

import sys
import glob
import csv
from pathlib import Path
from collections import defaultdict
import numpy as np

from counterexamples import is_correct_counterexample, CounterexampleResult
from settings import Settings, GnuplotSettings

class ToolResult:
    """Tool's result"""

    # columns
    CATEGORY = 0
    NETWORK = 1
    PROP = 2
    PREPARE_TIME = 3
    RESULT = 4
    RUN_TIME = 5

    all_categories = set()

    # stats
    num_verified = defaultdict(int) # number of benchmarks verified
    num_violated = defaultdict(int) 
    num_holds = defaultdict(int)
    incorrect_results = defaultdict(int)

    num_categories = defaultdict(int)
    toolerror_counts = defaultdict(int)

    def __init__(self, scored, tool_name, csv_path, cpu_benchmarks, skip_benchmarks):
        assert "csv" in csv_path

        self.tool_name = tool_name
        self.category_to_list = defaultdict(list) # maps category -> list of results

        self.skip_benchmarks = skip_benchmarks
        self.cpu_benchmarks = cpu_benchmarks
        self.gpu_overhead = np.inf # default overhead
        self.cpu_overhead = np.inf # if using separate overhead for cpu
        
        self.max_prepare = 0.0
        self.total_instances = dict()

        self.load(scored, csv_path)

    @staticmethod
    def reset():
        """reset static variables"""

        ToolResult.all_categories = set()

        # stats
        ToolResult.num_verified = defaultdict(int) # number of benchmarks verified
        ToolResult.num_violated = defaultdict(int) 
        ToolResult.num_holds = defaultdict(int)
        ToolResult.incorrect_results = defaultdict(int)

        ToolResult.num_categories = defaultdict(int)

        ToolResult.toolerror_counts = defaultdict(int)

    def result_instance_str(self, cat, index):
        """get a string representation of the instance for the given category and index"""

        row = self.category_to_list[cat][index]

        net = row[ToolResult.NETWORK]
        prop = row[ToolResult.PROP]

        return Path(net).stem + "-" + Path(prop).stem

    def single_result(self, cat, index):
        """get result_str, runtime of tool, after subtracting overhead"""

        row = self.category_to_list[cat][index]

        res = row[ToolResult.RESULT]
        t = float(row[ToolResult.RUN_TIME])

        # Overhead is no longer subtracted
        # t -= self.cpu_overhead if cat in self.cpu_benchmarks else self.gpu_overhead

        # prevent 0 times as this messes up log plots
        t = max(Settings.PLOT_MIN_TIME, t)

        return res, t

    def load(self, scored, csv_path):
        """load data from file"""

        unexpected_results = set()
                
        with open(csv_path, newline='') as csvfile:
            for row in csv.reader(csvfile):
                # rename results
                
                #print(f"{csv_path}: {row}")
                
                row[ToolResult.RESULT] = row[ToolResult.RESULT].lower()

                substitutions = Settings.CSV_SUBSTITUTIONS

                for from_prefix, to_str in substitutions:
                    if row[ToolResult.RESULT] == '': # don't use '' as prefix
                        row[ToolResult.RESULT] = 'unknown'
                    elif row[ToolResult.RESULT].startswith(from_prefix):
                        row[ToolResult.RESULT] = to_str

                network = row[ToolResult.NETWORK]
                result = row[ToolResult.RESULT]
                cat = row[ToolResult.CATEGORY]

                # in 2023, year was prepended to category
                year = ""

                assert "2025" in network, f"year not found in network path: {network}"
                year = "2025"

                cat = year + "_" + cat

                prepare_time = float(row[ToolResult.PREPARE_TIME])
                run_time = float(row[ToolResult.RUN_TIME])

                # workaround to drop convBigRELU from cifar2020
                if cat == 'cifar2020':
                    if 'convBigRELU' in network:
                        result = row[ToolResult.RESULT] = "unknown"

                if cat in self.skip_benchmarks or \
                        (scored and cat in Settings.UNSCORED_CATEGORIES) or \
                        (not scored and cat not in Settings.UNSCORED_CATEGORIES):
                    result = row[ToolResult.RESULT] = "unknown"

                if result.startswith('timeout'):
                    result = 'timeout' # fix for verapak "timeout(X_00 ..."

                if not ("test_nano" in network or "test_tiny" in network):
                    self.category_to_list[cat].append(row)
                    self.total_instances[cat] = self.total_instances.get(cat, 0) + 1

                if result not in ["holds", "violated", "timeout", "error", "unknown"]:
                    unexpected_results.add(result)
                    print(f"Unexpected results: {unexpected_results}")
                    exit(1)

                if result in ["holds", "violated"]:
                    if cat in self.cpu_benchmarks:
                        self.cpu_overhead = min(self.cpu_overhead, run_time)
                    else:
                        self.gpu_overhead = min(self.gpu_overhead, run_time)
                        
                    self.max_prepare = max(self.max_prepare, prepare_time)

        assert not unexpected_results, f"Unexpected results: {unexpected_results}"

        print(f"Loaded {self.tool_name}, default-overhead (gpu): {round(self.gpu_overhead, 1)}s," + \
              f"cpu-overhead: {round(self.cpu_overhead, 1)}s, " + \
              f"prepare time: {round(self.max_prepare, 1)}s")

        for skip_benchmark in self.skip_benchmarks:
            assert skip_benchmark in self.category_to_list, f"skip benchmark '{skip_benchmark}' not found in cat " + \
                f"list: {list(self.category_to_list.keys())}"

        self.delete_empty_categories()

    def delete_empty_categories(self):
        """delete categories without successful measurements"""

        to_remove = [] #['acasxu', 'cifar2020'] # benchmarks to skip

        for key in self.category_to_list.keys():
            rows = self.category_to_list[key]

            should_remove = True

            for row in rows:
                result = row[ToolResult.RESULT]

                if result in ('holds', 'violated'):
                    
                    should_remove = False
                    break

            if should_remove:
                to_remove.append(key)
            elif key != "test":
                ToolResult.all_categories.add(key)

        for key in to_remove:
            if key in self.category_to_list:
                #print(f"empty category {key} in tool {self.tool_name}")
                del self.category_to_list[key]

        ToolResult.num_categories[self.tool_name] = len(self.category_to_list)

class LongTableRow:
    """container object for longtable of results"""

    def __init__(self, cat: str, instance_id: int, result: str, tool_times_scores: Dict[str, Tuple[Union[str, float], int]]):
        self.cat = cat
        self.instance_id = instance_id

        assert result in ['sat', 'unsat', '-'], f"result was {result}"
        self.result = result
        self.tool_times_scores = tool_times_scores

def compare_results(all_tool_names, gnuplot_tool_cat_times, result_list, single_overhead, scored):
    """compare results across tools"""

    min_percent = 0 # minimum percent for total score

    total_score = defaultdict(int)
    all_cats = {}

    tool_times = {}

    longtable_data: List[LongTableRow] = []

    for tool in all_tool_names:
        tool_times[tool] = []

    if not ToolResult.all_categories:
        print(f"No categories selected when scored={scored}, skipping")
        return

    total_instances = dict()

    for cat in sorted(ToolResult.all_categories):
        print(f"\nCategory {cat}:")

        # maps tool_name -> [score, num_verified, num_falsified, num_fastest, num_errors]
        cat_score: Dict[str, List[int, int, int, int, int]] = {}
        all_cats[cat] = cat_score

        num_rows = 0

        participating_tools = []

        for tool_result in result_list:
            if cat in tool_result.total_instances:
                if cat in total_instances:
                    assert total_instances[cat] == tool_result.total_instances[cat], f"cat {cat} has different instance counts"
                else:
                    total_instances[cat] = tool_result.total_instances[cat]
            cat_dict = tool_result.category_to_list

            if not cat in cat_dict:
                continue
            
            rows = cat_dict[cat]
            assert num_rows == 0 or len(rows) == num_rows, f"tool {tool_result.tool_name}, cat {cat}, " + \
                f"got {len(rows)} rows expected {num_rows}"

            if num_rows == 0:
                num_rows = len(rows)
                print(f"Category {cat} has {num_rows} (from {tool_result.tool_name})")

            participating_tools.append(tool_result)

        # work with participating tools only
        tool_names = [t.tool_name for t in participating_tools]
        print(f"{len(participating_tools)} participating tools: {tool_names}")
        table_rows = []
        all_times = []
        all_results = []

        for index in range(num_rows):
            rand_gen_succeeded = False
            times_holds = []
            tools_holds = []
            times_violated = []
            tools_violated = []
            counterexamples_violated = []
            correct_violations = {}
            
            table_row = []
            table_rows.append(table_row)
            instance_str = participating_tools[0].result_instance_str(cat, index)
            table_row.append(instance_str)

            for t in participating_tools:
                res, secs = t.single_result(cat, index)

                if res == "unknown":
                    table_row.append("-")
                    continue

                if not res in ["holds", "violated"]:
                    table_row.append(res)
                    continue

                if res == "holds":
                    times_holds.append(secs)
                    tools_holds.append(t.tool_name)
                else:
                    assert res == "violated"
                    times_violated.append(secs)
                    tools_violated.append(t.tool_name)

                    # construct counterexample path
                    row = t.category_to_list[cat][index]
                    full_network_path = row[ToolResult.NETWORK]
                    net = Path(full_network_path).stem
                    prop = Path(row[ToolResult.PROP]).stem
                    
                    if "safenlp" in full_network_path:
                        if "medical" in full_network_path:
                            ce_path = f"../{t.tool_name}/{cat}/medical_{net}_{prop}.counterexample.gz"
                            net = f"medical/{net}"
                            prop = f"medical/{prop}"
                        else:
                            assert "ruarobot" in full_network_path
                            ce_path = f"../{t.tool_name}/{cat}/ruarobot_{net}_{prop}.counterexample.gz"
                            net = f"ruarobot/{net}"
                            prop = f"ruarobot/{prop}"
                    else:
                        ce_path = f"../{t.tool_name}/{cat}/{net}_{prop}.counterexample.gz"

                    if not Settings.SKIP_CE_FILES:
                        assert Path(ce_path).is_file(), f"CE path not found: {ce_path} and Settings.SKIP_CE_FILES is False"

                    tup = ce_path, cat, net, prop
                    counterexamples_violated.append(tup)

                table_row.append(f"{round(secs, 1)} ({res[0]})")

                if t.tool_name == "randgen":
                    assert res == "violated"
                    rand_gen_succeeded = True

            print()

            true_result = "-"

            if times_holds and not times_violated:
                true_result = 'unsat'
            elif times_violated and not times_holds:
                true_result = 'sat'


            if (Settings.ALWAYS_CHECK_COUNTEREXAMPLES and times_violated) or \
                (not Settings.ALWAYS_CHECK_COUNTEREXAMPLES and times_holds and times_violated):

                print(f"Checking counterexamplesfor index {index}. Violated: {len(times_violated)} " +
                      f"({tools_violated}), Holds: {len(times_holds)} ({tools_holds})")
                
                if times_holds and times_violated:
                    table_row.append('*multiple results*')

                for tup, tool in zip(counterexamples_violated, tools_violated):
                    print(f"\nchecking counterexample for {tool}")
                    res = is_correct_counterexample(*tup)

                    correct_violations[tool] = res

                print(f"were violated counterexamples valid?: {correct_violations}")

                if np.any([x == CounterexampleResult.CORRECT for x in correct_violations.values()]): ### HERE !!
                    true_result = 'sat'
                else:
                    true_result = 'unsat'

                # if times_holds and times_violated and np.all([x == CounterexampleResult.CORRECT_UP_TO_TOLERANCE for x in correct_violations.values()]):
                #     with open('conflicting.csv', 'a') as f:
                #         f.write(f"{cat},{index},{prop},{true_result},{tools_holds}, {tools_violated}," + \
                #                 f"{','.join([f'{t}={c}' for t, c in correct_violations.items()])}\n")
            print(f"Row: {table_row}")
            print(f"True Result: {true_result}")

            row_times = []
            all_times.append(row_times)
            all_results.append(None)
            tool_times_scores: Dict[str, Tuple[Union[str, float], int]] = {}
            
            for t in participating_tools:
                res, secs = t.single_result(cat, index)
                
                score, is_verified, is_falsified, is_fastest, is_error = get_score(t.tool_name, res, secs, rand_gen_succeeded,
                                                                times_holds, times_violated,
                                                                correct_violations)
                print(f"{index}: {t.tool_name} score: {score}, is_ver: {is_verified}, is_fals: {is_falsified}, " + \
                      f"is_fastest: {is_fastest}")

                if is_verified or is_falsified:
                    all_results[-1] = 'H' if is_verified else 'V'
                    row_times.append(secs)
                    
                    tool_times_scores[t.tool_name] = (secs, score)
                else:
                    row_times.append(None)

                    if is_error:
                        tool_times_scores[t.tool_name] = (secs, score)

                if t.tool_name in cat_score:
                    tool_score_tup = cat_score[t.tool_name]
                else:
                    tool_score_tup = [0, 0, 0, 0, 0]
                    cat_score[t.tool_name] = tool_score_tup

                # [score, num_verified, num_falsified, num_fastest]
                tool_score_tup[0] += score
                tool_score_tup[1] += 1 if is_verified else 0
                tool_score_tup[2] += 1 if is_falsified else 0
                tool_score_tup[3] += 1 if is_fastest else 0
                tool_score_tup[4] += 1 if is_error else 0
                tool_score_tup = None

            # accumulate long table data
            longtable_data.append(LongTableRow(cat, index, true_result, tool_times_scores))

        print("--------------------")
        num_holds = 0
        num_violated = 0
        num_unknown = 0

        for i, (row_times, result) in enumerate(zip(all_times, all_results)):
            assert len(row_times) == len(tool_names)

            if result is None:
                num_unknown += 1
            else:                
                for t, tool in zip(row_times, tool_names):
                    if t is not None:
                        #assert t > 0, "time was zero?"
                        tool_times[tool].append(t)
                        gnuplot_tool_cat_times[tool][cat].append(t)
                        gnuplot_tool_cat_times[tool]['all'].append(t)
                        if cat not in Settings.UNSCORED_CATEGORIES:
                            gnuplot_tool_cat_times[tool]['all_scored'].append(t)

                        print(f"!! {i}: tool={tool}, cat={cat}: time={t}")
                
                if result == 'V':
                    num_violated += 1
                elif result == 'H':
                    num_holds += 1
        
        print(f"Total Violated: {num_violated}")
        print(f"Total Holds: {num_holds}")
        print(f"Total Unknown: {num_unknown}")
        

        print("--------------------")
        print(", ".join(tool_names))

        for table_row in table_rows:
            print(", ".join(table_row))

        print(f"---------\nCategory {cat}:")

        if cat_score:
            max_score = max([t[0] for t in cat_score.values()])

            for tool, score_tup in cat_score.items():
                score = score_tup[0]
                if max_score > 0:
                    percent = max(min_percent, 100 * score / max_score)
                else:
                    percent = 0.0
                print(f"{tool}: {score} ({round(percent, 2)}%)")

                total_score[tool] += percent

    print("\n###############")
    print("### Summary ###")
    print("###############")
    
    sorted_tools = []

    with open(Settings.TOTAL_SCORE_LATEX, 'w', encoding='utf-8') as f:
        tee(f, "\n%Total Score:")
        res_list = []

        print_table_header(f, "Overall Score", "tab:score", ["\\# ~", "Tool", "Score"])

        for tool, score in total_score.items():
            tool_latex = latex_tool_name(tool)
            desc = f"{tool_latex} & {round(score, 1)} \\\\"

            res_list.append((score, desc, tool))

        for i, s in enumerate(reversed(sorted(res_list))):
            sorted_tools.append(s[2])
            
            tee(f, f"{i+1} & {s[1]}")

        print_table_footer(f)

        add_image(f, f'all')
        add_image(f, f'all_scored')

    #######
    write_gnuplot_files(gnuplot_tool_cat_times, sorted_tools)
    #######

    print("--------------------")

    for cat in sorted(all_cats.keys()):
        cat_score = all_cats[cat]

        if not cat_score:
            continue

        filename = Settings.UNSCORED_LATEX if cat in Settings.UNSCORED_CATEGORIES else Settings.SCORED_LATEX

        with open(filename, 'a', encoding='utf-8') as f:
        
            tee(f, f"\n\\clearpage\n% Category {cat} (single_overhead={single_overhead}):")
            res_list = []
            max_score = max([t[0] for t in cat_score.values()])

            cat_str = cat.replace('_', '-')

            print_table_header(f, f"Benchmark \\texttt{{{cat_str}}}", f"tab:cat_{cat}",
                               ("\\# ~", "Tool", "Verified", "Falsified", "Fastest", "Penalty", "Points", "Score", "Solved"),
                               align='llllllrrr')

            for tool, score_tup in cat_score.items():
                score, num_verified, num_falsified, num_fastest, num_error = score_tup

                if max_score > 0:
                    percent = max(min_percent, 100 * score / max_score)
                else:
                    percent = 0.0
                tool_latex = latex_tool_name(tool)

                #desc = f"{tool}: {score} ({round(percent, 2)}%)"
                desc = f"{tool_latex} & {num_verified} & {num_falsified} & {num_fastest} & {num_error} & {score} & {round(percent, 1)} & {(num_verified + num_falsified) * 100.0 / total_instances[cat]:.1f}\\% \\\\"

                res_list.append((score, desc))

            for i, s in enumerate(reversed(sorted(res_list))):
                tee(f, f"{i+1} & {s[1]}")

            print_table_footer(f)

            add_image(f, cat)

    ################
    # print longtable_data
            
    with open(Settings.LONGTABLE_LATEX, 'w', encoding='utf-8') as f:
        tee(f, f"% Long table of all results\n\n")

        num_tools = len(sorted_tools)

        headers = ("Category", "Id", "Result") + tuple(f"{longtable_tool_name(t)}" for t in sorted_tools)

        caption = "Instance Runtimes. Fastest times are \\textcolor{blue}{blue}. "
        caption += "Second fastest are \\textcolor{second}{green}. Penalties are red crosses (" +\
          f"\\textbf{{\\textcolor{{red}}{{\\ding{{55}}}}}}" + ")."

        print_longtable_header(f, caption,  "tab:all_results", headers)

        last_cat = None

        for ltd in longtable_data:

            if ltd.cat != last_cat:
                if last_cat != None:
                    tee(f, "\\midrule")
                    
                last_cat = ltd.cat

            tool_results = ""
            for tool_index, tool in enumerate(sorted_tools):

                if tool_index > 0:
                    tool_results += " & "
                    
                if tool in ltd.tool_times_scores:
                    t, score = ltd.tool_times_scores[tool]

                    if isinstance(t, str):
                        tool_results += t
                    else:
                        if score == 12:
                            color = "blue"
                        elif score == 11:
                            # \definecolor{second}{HTML}{3C8031}
                            color = "second"
                        elif score == 10:
                            color = "darkgray"
                        elif score < 0:
                            color = "red"

                        if score < 0:
                            # \ding{55} is from package pifont
                            tool_results += f"~~\\textbf{{\\textcolor{{{color}}}{{\\ding{{55}}}}}}"
                        else:
                            tool_results += f"\\textcolor{{{color}}}{{{round_time(t)}}}"
                else:
                    tool_results += "-"

            true_result = ltd.result

            # override true result manually
            for prefix, index, new_result in Settings.OVERRIDE_RESULTS:
                if ltd.cat.startswith(prefix) and ltd.instance_id == index:
                    true_result = new_result
            
            pretty_res = f"~\\textsc{{{true_result}}}" if ltd.result != "-" else "~?"
            
            tee(f, f"{latex_cat_name(ltd.cat)} & {ltd.instance_id} & {pretty_res} & {tool_results} \\\\")

        print_longtable_footer(f)

def round_time(t):
    """round time in table"""

    if t >= 99.9:
        rv = f"{t:.0f}"
    elif t < 0.01:
        rv = "$<$0.01"
    elif t >= 10:
        rv = f"{t:.1f}"
    else:
        rv = f"{t:.2f}"

    return rv

def add_image(fout, prefix):
    """add latex code for an image with the given prefix.pdf"""

    title = "Unknown"

    for gps in Settings.gnuplot_data:
        if gps.prefix == prefix:
            title = gps.title

    tee(fout, """
\\begin{figure}[h]
\\centerline{\\includegraphics[width=\\textwidth]{""" + f"{Settings.PLOT_FOLDER}/{prefix}" + """.pdf}}
\\caption{Cactus Plot for """ + title + """.}
\\label{fig:quantPic}
\\end{figure}
""")

def tee(fout, line):
    """print to temrinal and a file"""
    
    print(line)
    fout.write(line + "\n")

def print_table_header(f, title, label, columns, align=None):
    """print latex table header"""

    bold_columns = ["\\textbf{" + c + "}" for c in columns]

    if align is None:
        align = 'l' * len(columns)
    else:
        assert len(columns) == len(align)

    tee(f, '\n\\begin{table}[h]')
    tee(f, '\\begin{center}')
    tee(f, '\\caption{' + title + '} \\label{' + label + '}')
    tee(f, '{\\setlength{\\tabcolsep}{2pt}')
    tee(f, '\\begin{tabular}[h]{@{}' + align + '@{}}')
    tee(f, '\\toprule')
    tee(f, ' & '.join(bold_columns) + "\\\\")
    #\textbf{\# ~} & \textbf{Tool} & \textbf{Score}  \\
    tee(f, '\\midrule')

def print_longtable_header(f, title, label, columns, align=None):
    """print latex table header"""

    bold_columns = ["\\textbf{" + c + "}" for c in columns]

    if align is None:
        align = 'l' * len(columns)
    else:
        assert len(columns) == len(align)

    tee(f, '''\\begin{center}
{\\setlength{\\tabcolsep}{1pt}
\\scriptsize
\\begin{longtable}{@{}''' + align + '''@{}}''')
    
    tee(f, '\\caption{\\footnotesize ' + title + '} \\label{' + label + '} \\\\')
    #tee(f, '\\caption{\\footnotesize ' + title + '} \\\\')
    tee(f, '\\toprule')
    tee(f, ' & '.join(bold_columns) + " \\\\")
    #\textbf{\# ~} & \textbf{Tool} & \textbf{Score}  \\
    tee(f, '\\midrule')
    tee(f, '\\endhead')

def print_table_footer(f):
    """print latex table footer"""

    tee(f, '''\\bottomrule
\\end{tabular}
}
\\end{center}
\\end{table}\n\n''')

def print_longtable_footer(f):
    """print latex longtable footer"""

    tee(f, '''\\bottomrule
\end{longtable}
}
\end{center}\n\n''')


def get_score(tool_name, res, secs, rand_gen_succeded, times_holds, times_violated, ce_results):
    """Get the score for the given result
    Actually returns a 4-tuple: score, is_verified, is_falsified, is_fastest

    Correct hold: 10 points
    Correct violated (where random tests did not succeed): 10 points
    Correct violated (where random test succeeded): 1 point
    Incorrect result: Settings.PENALTY_INCORRECT points

    Time bonus: 
        The fastest tool for each solved instance will receive +2 points. 
        The second fastest tool will receive +1 point.
        If two tools have runtimes within 0.2 seconds, we will consider them the same runtime.
    """

    penalize_no_ce = True

    is_verified = False
    is_falsified = False
    is_fastest = False
    is_error = False

    num_holds = len(times_holds)
    num_violated = len(times_violated)

    #print(f"tool: {tool_name} {res}")

    valid_ce_any_tool = False
    valid_ce_this_tool = False

    for ce_tool_name, ce_valid_res in ce_results.items():
        # The ce may be within the tolerance, but outside the real bounds.
        # In that case, do not penalize this tool, but also do not assume this instance
        # is SAT.

        if ce_valid_res == CounterexampleResult.CORRECT:
            valid_ce_any_tool = True
        if ce_tool_name == tool_name and ce_valid_res in [CounterexampleResult.CORRECT, CounterexampleResult.CORRECT_UP_TO_TOLERANCE]:
            valid_ce_this_tool = True

    assert not rand_gen_succeded, "VNNCOMP doesn't use randgen anymore"
    correct_result = False

    if res not in ["holds", "violated"]:
        score = 0
    elif res == "violated" and not tool_name in ce_results: # didn't check counterexample due to settings
        correct_result = True
    elif penalize_no_ce and res == "violated" and not ce_results[tool_name]: # in 2022 also had: num_holds > 0 
        # Rule: If a witness is not provided when “sat” is produced, the tool will be assessed a penalty.
        score = Settings.PENALTY_INCORRECT
        ToolResult.incorrect_results[tool_name] += 1
        print(f"tool {tool_name} did not produce a valid counterexample and there are mismatching results")

        ToolResult.toolerror_counts[f'{tool_name}_no-ce-but-required'] += 1
        is_error = True
    elif res == "violated" and not valid_ce_this_tool:
        # incorrect witness
        score = Settings.PENALTY_INCORRECT
        ToolResult.incorrect_results[tool_name] += 1
        is_error = True

        ToolResult.toolerror_counts[f'{tool_name}_{ce_results[tool_name]}'] += 1
    elif res == "holds" and valid_ce_any_tool:
        score = Settings.PENALTY_INCORRECT
        ToolResult.incorrect_results[tool_name] += 1
        is_error = True

        ToolResult.toolerror_counts[f'{tool_name}_incorrect_unsat'] += 1
    else:
        correct_result = True
    
    if correct_result:

        ToolResult.num_verified[tool_name] += 1

        if res == "holds":
            is_verified = True
            times = times_holds.copy()
            ToolResult.num_holds[tool_name] += 1
        else:
            assert res == "violated"
            times = times_violated.copy()
            ToolResult.num_violated[tool_name] += 1

            is_falsified = True
            
        score = 10

        add_time_bonus = False
        
        if add_time_bonus:
            clamped_times = [max(t, Settings.SCORING_MIN_TIME) for t in times]
            secs = max(secs, Settings.SCORING_MIN_TIME)

            min_time = min(clamped_times)

            if secs < min_time + 0.2:
                score += 2
                is_fastest = True
            else:
                clamped_times.remove(min_time)
                second_time = min(clamped_times)

                if secs < second_time + 0.2:
                    score += 1

    return score, is_verified, is_falsified, is_fastest, is_error

def print_stats(result_list):
    """print stats about measurements"""

    with open(Settings.STATS_LATEX, 'w', encoding='utf-8') as f:
        tee(f, '\n%%%%%%%%%% Stats %%%%%%%%%%%')

        tee(f, "\n% Overhead:")
        olist = []

        for r in result_list:
            olist.append((r.gpu_overhead, r.cpu_overhead, r.tool_name))

        #print_table_header("Overhead", "tab:overhead", ["\\# ~", "Tool", "Seconds", "~~CPU Mode"], align='llrr')
        print_table_header(f, "Overhead", "tab:overhead", ["\\# ~", "Tool", "Seconds"], align='llr')

        for i, n in enumerate(sorted(olist)):
            #cpu_overhead = "-" if n[1] == np.inf else round(n[1], 1)

            #print(f"{i+1} & {n[2]} & {round(n[0], 1)} & {cpu_overhead} \\\\")
            tee(f, f"{i+1} & {latex_tool_name(n[2])} & {round(n[0], 1)} \\\\")

        print_table_footer(f)

        items = [("Num Benchmarks Participated", ToolResult.num_categories),
                 ("Num Instances Verified", ToolResult.num_verified),
                 ("Num SAT", ToolResult.num_violated),
                 ("Num UNSAT", ToolResult.num_holds),
                 ("Incorrect Results (or Missing CE)", ToolResult.incorrect_results),
                 ]

        for index, (label, d) in enumerate(items):
            tee(f, f"\n% {label}:")

            tab_label = f"tab:stats{index}"
            print_table_header(f, label, tab_label, ["\\# ~", "Tool", "Count"], align='llr')

            l = []

            for tool, count in d.items():
                tool_latex = latex_tool_name(tool)

                l.append((count, tool_latex))

            for i, s in enumerate(reversed(sorted(l))):
                tee(f, f"{i+1} & {s[1]} & {s[0]} \\\\")

            print_table_footer(f)

    print(ToolResult.toolerror_counts)

def latex_cat_name(cat):
    """get latex version of category name"""

    subs = Settings.CAT_NAME_SUBS_LATEX
    found = False

    for old, new in subs:
        if cat == old:
            cat = new
            found = True
            break

    if not found:
        cat = cat.replace("_", " ")
        cat = ' '.join(e.capitalize() for e in cat.split())

    return cat

def longtable_tool_name(tool):
    """get latex version of tool name"""

    subs = Settings.TOOL_NAME_SUBS_LONGTABLE

    found = False

    for old, new in subs:
        if tool == old:
            tool = new
            found = True
            break

    #if not found:
    #    tool = tool.capitalize()

    return tool

def latex_tool_name(tool):
    """get latex version of tool name"""

    subs = Settings.TOOL_NAME_SUBS_LATEX

    found = False

    for old, new in subs:
        if tool == old:
            tool = new
            found = True
            break

    if not found:
        tool = tool.capitalize()

    return tool

def gnuplot_tool_name(tool):
    """get fnuplot version of tool name"""

    subs = Settings.TOOL_NAME_SUBS_GNUPLOT

    found = False

    for old, new in subs:
        if tool == old:
            tool = new
            found = True
            break

    if not found:
        tool = tool.capitalize()

    return tool

def write_gnuplot_files(gnuplot_tool_cat_times, sorted_tools):
    """write files with data for gnuplot cactus plots"""

    for gps in Settings.gnuplot_data:
        cat = gps.prefix
        
        for tool in gnuplot_tool_cat_times.keys():
            #assert cat in gnuplot_tool_cat_times[tool], f"cat {cat} not in {tool}: {gnuplot_tool_cat_times[tool].keys()}"
            times_list = gnuplot_tool_cat_times[tool][cat]
        
            times_list.sort()

            with open(Settings.PLOTS_DIR + f"/accumulated-{cat}-{tool}.txt", 'w', encoding='utf-8') as f:
                for i, t in enumerate(times_list):
                    f.write(f"{t}\t{i+1}\n")

    with open(Settings.PLOTS_DIR + "/generated.gnuplot", 'w', encoding='utf-8') as f:
        #########################
        # input_list
        f.write("input_list = \"")

        for gps in Settings.gnuplot_data:
            cat = gps.prefix

            f.write("'")

            for tool in sorted_tools:

                times_list = gnuplot_tool_cat_times[tool][cat]

                if times_list:
                    f.write(f"{cat}-{tool} ")

            f.write("' ")

        f.write("\"\n\n")

        #########################
        # pretty_input_list
        f.write("pretty_input_list = \"")
        
        for gps in Settings.gnuplot_data:
            cat = gps.prefix

            f.write("\\\"")

            # sort tools by category

            for tool in sorted_tools:

                times_list = gnuplot_tool_cat_times[tool][cat]

                if times_list:
                    f.write(f"'{gnuplot_tool_name(tool)}' ")

            f.write("\\\" ")

        f.write("\"\n\n")

        #########################
        # tool_index

        f.write("tool_index_list = \"")
        
        for gps in Settings.gnuplot_data:
            cat = gps.prefix

            f.write("'")

            # sort tools by category

            for i, tool in enumerate(sorted_tools):

                times_list = gnuplot_tool_cat_times[tool][cat]

                if times_list:
                    f.write(f"{i} ")

            f.write("' ")

        f.write("\"\n\n")
        
        ##########################
        # title_list

        f.write("title_list = \"")

        for gps in Settings.gnuplot_data:
            f.write(f"'{gps.title}' ")

        f.write("\"\n\n")

        ##########################
        # outputs

        f.write("outputs = '")

        for i,  gps in enumerate(Settings.gnuplot_data):
            f.write(f"{gps.prefix}.pdf ")

        f.write("'\n\n")

        #########################
        # xmax_plot_list

        f.write("xmax_plot_list = '")
        
        for gps in Settings.gnuplot_data:
            cat = gps.prefix

            # sort tools by category
            max_times = 0

            for tool in sorted_tools:
                times_list = gnuplot_tool_cat_times[tool][cat]

                if len(times_list) > max_times:
                    max_times = len(times_list)

            f.write(f"{1.05 * max_times} ")

        f.write("'\n\n")

        #########################
        # ymin_list

        f.write(f"ymin_list = '")
        count = 10

        for gps in Settings.gnuplot_data:
            cat = gps.prefix

            all_times = []
            
            for tool in sorted_tools:
                all_times += gnuplot_tool_cat_times[tool][cat]

            all_times = np.array(all_times)
            
            if np.sum(all_times < 0.1) > count:
                min_time = 0.8 * 0.01
            elif np.sum(all_times < 1.0) > count:
                min_time = 0.8 * 0.1
            else:
                min_time = 0.8 * 1.0

            f.write(f"{round(min_time, 4)} ")

        assert min_time > 0



        f.write("'\n\n")

        #########################
        # timeout_y_list

        f.write("timeout_y_list = '")
        
        for gps in Settings.gnuplot_data:
            cat = gps.prefix

            # sort tools by category
            max_time = 0

            for tool in sorted_tools:
                times_list = gnuplot_tool_cat_times[tool][cat]

                if times_list and times_list[-1] > max_time:
                    max_time = times_list[-1]

            if max_time > 300:
                f.write("300 ")
            else:
                f.write("60 ")

        f.write("'\n\n")

        #########################
        # timeout_str_list

        f.write("timeout_str_list = \"")
        
        for gps in Settings.gnuplot_data:
            cat = gps.prefix

            # sort tools by category
            max_time = 0

            for tool in sorted_tools:
                times_list = gnuplot_tool_cat_times[tool][cat]

                if times_list and times_list[-1] > max_time:
                    max_time = times_list[-1]

            if max_time > 300:
                f.write("'Five Minutes' ")
            else:
                f.write("'One Minute' ")

        f.write("\"\n\n")

        #########################
        # timeout_x_list

        f.write("timeout_x_list = '")
        
        for gps in Settings.gnuplot_data:
            cat = gps.prefix

            # sort tools by category
            max_times = 0

            for tool in sorted_tools:
                times_list = gnuplot_tool_cat_times[tool][cat]

                if len(times_list) > max_times:
                    max_times = len(times_list)

            max_times = 1.05 * max_times
            f.write(f"{1 + 0.01 * max_times} ")

        f.write("'\n\n")

        #########################
        # ymax_list

        f.write("ymax_list = '")
        
        for gps in Settings.gnuplot_data:
            cat = gps.prefix

            # sort tools by category
            max_time = 0

            for tool in sorted_tools:
                times_list = gnuplot_tool_cat_times[tool][cat]

                if times_list and times_list[-1] > max_time:
                    max_time = times_list[-1]

            f.write(f"{1.5*max_time} ")

        f.write("'\n\n")

        #########################
        # key_list

        f.write("key_list = \"")
        
        for gps in Settings.gnuplot_data:
            cat = gps.prefix

            # sort tools by category
            max_instances = 0
            max_time = 0

            for tool in sorted_tools:
                times_list = gnuplot_tool_cat_times[tool][cat]
                    
                if len(times_list) > max_instances:
                    max_instances = len(times_list)

                if times_list and times_list[-1] > max_time:
                    max_time = times_list[-1]

            xplot_limit = 1.07 * max_instances
            yplot_limit = 1.5 * max_time

            f.write(f"'{1.05 * xplot_limit} {yplot_limit}' ")

        f.write("\"\n\n")

def process_single_tool_or_benchmark(csv_path):
    """
    Process results from a single tool's results.csv file or a specific benchmark's results.
    
    Args:
        csv_path: Path to the results.csv file, can be either:
                 - {tool}/results.csv (for all benchmarks of a tool)
                 - {tool}/{benchmark}/results.csv (for a specific benchmark)
    """
    # Extract tool name and possibly benchmark name from path
    path_parts = csv_path.split('/')
    tool_name = path_parts[Settings.TOOL_LIST_GLOB_INDEX]
    
    # Determine if this is a specific benchmark or all benchmarks
    is_benchmark_specific = len(path_parts) > Settings.TOOL_LIST_GLOB_INDEX + 2 and path_parts[-1] == "results.csv"
    
    if is_benchmark_specific:
        benchmark_name = path_parts[Settings.TOOL_LIST_GLOB_INDEX + 1]
        log_file_path = f"results_{tool_name}_{benchmark_name}.log"
        print(f"Processing specific benchmark '{benchmark_name}' for tool '{tool_name}'")
    else:
        log_file_path = f"results_{tool_name}_all.log"
        print(f"Processing all benchmarks for tool '{tool_name}'")
    
    # Open the log file
    with open(log_file_path, 'w', encoding='utf-8') as log_file:
        def log_print(*args, **kwargs):
            # Print to console
            print(*args, **kwargs)
            # Print to log file
            print(*args, file=log_file, **kwargs)
            
        if is_benchmark_specific:
            log_print(f"\n===== Processing single benchmark result of a tool: {csv_path} =====\n")
            log_print(f"Tool name: {tool_name}")
            log_print(f"Benchmark name: {benchmark_name}")
        else:
            log_print(f"\n===== Processing single tool results: {csv_path} =====\n")
            log_print(f"Tool name: {tool_name}")
        
        # Create a ToolResult object to process the CSV
        cpu_benchmarks = []
        skip_benchmarks = []
        
        # Process both scored and unscored categories
        benchmark_results = {}
        
        for scored in [False, True]:
            tr = ToolResult(scored, tool_name, csv_path, cpu_benchmarks, skip_benchmarks)
            
            # Group results by category
            for cat, instances in tr.category_to_list.items():
                if cat not in benchmark_results:
                    benchmark_results[cat] = []
                benchmark_results[cat].extend(instances)
        
        # Print results by category
        log_print(f"\nResults for tool: {tool_name}")
        log_print("=" * 80)
        
        total_holds = 0
        total_violated = 0
        total_timeout = 0
        total_error = 0
        total_unknown = 0
        total_ce_correct = 0
        total_ce_incorrect = 0
        total_ce_missing = 0
        
        for cat in sorted(benchmark_results.keys()):
            instances = benchmark_results[cat]
            
            category_holds = 0
            category_violated = 0
            category_timeout = 0
            category_error = 0
            category_unknown = 0
            category_ce_correct = 0
            category_ce_incorrect = 0
            category_ce_missing = 0
            
            log_print(f"\nCategory: {cat} ({len(instances)} instances)")
            log_print("-" * 80)
            log_print(f"{'Instance':40} {'Result':10} {'Time (s)':10} {'CE Status':15}")
            log_print("-" * 80)
            
            for instance_idx, row in enumerate(instances, 1):

                # Add instance counter before each instance
                log_print(f"--- INSTANCE {instance_idx}/{len(instances)} ---")
                
                network = Path(row[ToolResult.NETWORK]).stem
                prop = Path(row[ToolResult.PROP]).stem
                full_network_path = row[ToolResult.NETWORK]
                instance = f"{network}-{prop}"
                result = row[ToolResult.RESULT]
                runtime = float(row[ToolResult.RUN_TIME])
                
                ce_status = ""
                
                # Check counterexample for "violated" (sat) results
                if result == "violated":
                    # Construct counterexample path
                    if "safenlp" in full_network_path:
                        if "medical" in full_network_path:
                            ce_path = f"../{tool_name}/{cat}/medical_{network}_{prop}.counterexample.gz"
                            net = f"medical/{network}"
                            prop_name = f"medical/{prop}"
                        else:
                            # Assuming "ruarobot" as in the original code
                            ce_path = f"../{tool_name}/{cat}/ruarobot_{network}_{prop}.counterexample.gz"
                            net = f"ruarobot/{network}"
                            prop_name = f"ruarobot/{prop}"
                    else:
                        ce_path = f"../{tool_name}/{cat}/{network}_{prop}.counterexample.gz"
                        net = network
                        prop_name = prop
                    
                    try:
                        # Validate counterexample
                        tup = ce_path, cat, net, prop_name
                        res = is_correct_counterexample(*tup)
                        
                        # Check if the counterexample is valid
                        if res == CounterexampleResult.CORRECT:
                            ce_status = "VALID"
                            category_ce_correct += 1
                            total_ce_correct += 1
                        elif res == CounterexampleResult.CORRECT_UP_TO_TOLERANCE:
                            ce_status = "VALID (TOL)"
                            category_ce_correct += 1
                            total_ce_correct += 1
                        else:
                            ce_status = f"INVALID ({res})"
                            log_print(f"  - CE Error: {res} for {instance}")
                            category_ce_incorrect += 1
                            total_ce_incorrect += 1
                            
                    except FileNotFoundError:
                        ce_status = "MISSING"
                        log_print(f"  - Missing CE: {ce_path} for {instance}")
                        category_ce_missing += 1
                        total_ce_missing += 1
                    except Exception as e:
                        ce_status = f"ERROR: {str(e)[:20]}"
                        log_print(f"  - CE Error: {str(e)} for {instance}")
                        category_ce_incorrect += 1
                        total_ce_incorrect += 1
                
                log_print(f"{instance:40} {result:10} {runtime:10.2f} {ce_status:15}")
                
                # Update counters
                if result == "holds":
                    category_holds += 1
                    total_holds += 1
                elif result == "violated":
                    category_violated += 1
                    total_violated += 1
                elif result == "timeout":
                    category_timeout += 1
                    total_timeout += 1
                elif result == "error":
                    category_error += 1
                    total_error += 1
                else:
                    category_unknown += 1
                    total_unknown += 1
            
            # Print category summary
            log_print("-" * 80)
            log_print(f"Category Summary: holds={category_holds}, violated={category_violated}, "
                  f"timeout={category_timeout}, error={category_error}, unknown={category_unknown}")
            if category_violated > 0:
                log_print(f"Counterexample Summary: valid={category_ce_correct}, invalid={category_ce_incorrect}, "
                      f"missing={category_ce_missing}")
        
        # Print overall summary
        log_print("\n" + "=" * 80)
        log_print(f"Overall Summary for {tool_name}:")
        log_print(f"Total categories: {len(benchmark_results)}")
        log_print(f"Total instances: {total_holds + total_violated + total_timeout + total_error + total_unknown}")
        log_print(f"  - holds:   {total_holds}")
        log_print(f"  - violated: {total_violated}")
        log_print(f"  - timeout:  {total_timeout}")
        log_print(f"  - error:    {total_error}")
        log_print(f"  - unknown:  {total_unknown}")
        if total_violated > 0:
            log_print(f"Counterexample Summary:")
            log_print(f"  - valid:   {total_ce_correct} ({total_ce_correct/total_violated*100:.1f}%)")
            log_print(f"  - invalid: {total_ce_incorrect} ({total_ce_incorrect/total_violated*100:.1f}%)")
            log_print(f"  - missing: {total_ce_missing} ({total_ce_missing/total_violated*100:.1f}%)")
        log_print("=" * 80)
        log_print(f"\nLog saved to: {log_file_path}")

def main():
    """main entry point"""
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Process VNN-COMP results')
    parser.add_argument('--single-tool', '-t', type=str, help='Process a single tool\'s results.csv file')
    parser.add_argument('--single-benchmark', '-b', type=str, help='Process a results.csv file of a single benchmark of an specific tool')
    args = parser.parse_args()

    # If single tool mode is requested, process only that tool
    if args.single_tool:
        if Path(args.single_tool).exists():
            process_single_tool_or_benchmark(args.single_tool)
            return
        else:
            print(f"Error: Results file not found: {args.single_tool}")
            sys.exit(1)
    
     # If single tool mode is requested, process only that tool
    if args.single_benchmark:
        if Path(args.single_benchmark).exists():
            process_single_tool_or_benchmark(args.single_benchmark)
            return
        else:
            print(f"Error: Results file not found: {args.single_benchmark}")
            sys.exit(1)

    # use single overhead for all tools. False will have two different overheads for some tools depending
    # on if GPU needed to be initialized (manually entered)
    single_overhead = True
    print(f"using single_overhead={single_overhead}")

    #####################################3
    #csv_list = glob.glob("results_csv/*.csv")
    csv_list = glob.glob(Settings.CSV_GLOB)
    #csv_list = ["../fastbatllnn/results.csv", "../nnenum/results.csv", "../alpha_beta_crown/results.csv"]
    print("!! using hardcoded csv list")

    csv_list.sort()

    assert csv_list, "no csv files found with glob: " + Settings.CSV_GLOB
  
    # clear files so we can append to them
    with open(Settings.SCORED_LATEX, 'w', encoding='utf-8') as f:
        pass

    with open(Settings.UNSCORED_LATEX, 'w', encoding='utf-8') as f:
        pass

    if Settings.SKIP_TOOLS:
        new_csv_list = []

        for csv_file in csv_list:
            skip_tool = False
            
            for skip in Settings.SKIP_TOOLS:
                if skip in csv_file:
                    skip_tool = True
                    break

            if not skip_tool:
                new_csv_list.append(csv_file)

            csv_list = new_csv_list

    tool_list = [c.split('/')[Settings.TOOL_LIST_GLOB_INDEX] for c in csv_list]

    cpu_benchmarks = {x: [] for x in tool_list}
    skip_benchmarks = {x: [] for x in tool_list}
    #skip_benchmarks['RPM'] = ['mnistfc']

    for tool, benchmark in Settings.SKIP_BENCHMARK_TUPLES:
        assert tool in tool_list, f"{tool} not in tool list: {tool_list}"
        skip_benchmarks[tool].append(benchmark)

    if not single_overhead: # Define a dict with the cpu_only benchmarks for each tool
        #pass
        cpu_benchmarks["ERAN"] = ["acasxu", "eran"]

    gnuplot_tool_cat_times = {} # accumulate for both scored and unscored

    for tool in tool_list:
        gnuplot_tool_cat_times[tool] = defaultdict(list)
        
    for scored in [False, True]:
        result_list = []
        ToolResult.reset()

        for csv_path, tool_name in zip(csv_list, tool_list):
            if tool_name.lower() == "scoring":
                continue
            tr = ToolResult(scored, tool_name, csv_path, cpu_benchmarks[tool_name], skip_benchmarks[tool_name])
            result_list.append(tr)

        # compare results across tools
        compare_results(tool_list, gnuplot_tool_cat_times, result_list, single_overhead, scored)

        if scored:
            print_stats(result_list)

    if Settings.SKIP_TOOLS:
        print(f"Note: tools were skipped: {Settings.SKIP_TOOLS}")

    if Settings.SKIP_CE_FILES:
        print(f"Note: CE file checking was skipped")

if __name__ == "__main__":
    #from counterexamples import get_ce_diff
    #get_ce_diff.clear_cache()

    main()
