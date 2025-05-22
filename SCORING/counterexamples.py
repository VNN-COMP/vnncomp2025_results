"""
code related to checking for counterexamples
"""

from pathlib import Path
import gzip
import datetime

import numpy as np
import onnx
import onnxruntime as ort

# Prevent a bug in some environments
# https://github.com/microsoft/onnxruntime/issues/8313#issuecomment-1486097717
_default_session_options = ort.capi._pybind_state.get_default_session_options()
def get_default_session_options_new():
     _default_session_options.inter_op_num_threads = 1
     _default_session_options.intra_op_num_threads = 1
     return _default_session_options
ort.capi._pybind_state.get_default_session_options = get_default_session_options_new

from vnnlib import read_vnnlib_simple, get_io_nodes

from cachier import cachier
from settings import Settings

def predict_with_onnxruntime(model_def, *inputs):
    'run an onnx model'

    sess_opt = ort.SessionOptions()
    sess_opt.intra_op_num_threads = 12
    sess_opt.inter_op_num_threads = 12
    sess = ort.InferenceSession(model_def.SerializeToString(), sess_opt)
    names = [i.name for i in sess.get_inputs()]

    inp = dict(zip(names, inputs))
    res = sess.run(None, inp)

    #names = [o.name for o in sess.get_outputs()]

    return res[0]

def read_ce_file(ce_path):
    """get file contents"""

    if ce_path.endswith('.gz'):
        with gzip.open(ce_path, 'rb') as f:
            content = f.read().decode('utf-8')
    else:
        with open(ce_path, 'r', encoding='utf-8') as f:
            content = f.read()

    content = content.replace('\n', ' ').strip()

    return content

class CounterexampleResult:
    """enum for return value of is_correct_counterexample"""

    CORRECT = "correct"
    CORRECT_UP_TO_TOLERANCE = "correct_up_to_tolerance"
    NO_CE = "no_ce"
    EXEC_DOESNT_MATCH = "exec_doesnt_match"
    SPEC_NOT_VIOLATED = "spec_not_violated"

def is_correct_counterexample(ce_path, cat, net, prop):
    """is the counterexample correct? returns an element of CounterexampleResult 
    """

    print(f"Checking ce path: {ce_path}, {cat}")

    benchmark_repo = ""

    for key in Settings.BENCHMARK_REPOS:
        if key != "2024":
            print("Skip ", key)
            continue
        print("ok", ce_path)
        if key in ce_path:
            benchmark_repo = Settings.BENCHMARK_REPOS[key]
            break

    assert benchmark_repo, f"benchmark directory (from Settings.BENCHMARK_REPOS={Settings.BENCHMARK_REPOS.keys()}) not found for ce path: {ce_path}"
 
    assert "_" == cat[4], f"expected year at start of cat: {cat}"
    cat_no_year = cat[5:]
    onnx_filename = f"{benchmark_repo}/benchmarks/{cat_no_year}/onnx/{net}.onnx"
    vnnlib_filename = f"{benchmark_repo}/benchmarks/{cat_no_year}/vnnlib/{prop}.vnnlib"

    if not Path(onnx_filename).is_file():
        # try unzipping
        gz_path = f"{onnx_filename}.gz"

        if not Path(gz_path).is_file():
            print(f"WARNING: onnx and gz path don't exist: {gz_path}")
        else:
            print(f"extracting from {gz_path} to {onnx_filename}")
            
            with gzip.open(gz_path, 'rb') as f:
                content = f.read()

                with open(onnx_filename, 'wb') as fout:
                    fout.write(content)

    if not Path(vnnlib_filename).is_file():
        # try unzipping
        gz_path = f"{vnnlib_filename}.gz"

        if Path(gz_path).is_file():
            print(f"extracting from {gz_path} to {vnnlib_filename}")
            
            with gzip.open(gz_path, 'rb') as f:
                content = f.read()

                with open(vnnlib_filename, 'wb') as fout:
                    fout.write(content)

    assert Path(onnx_filename).is_file(), f"onnx file '{onnx_filename}' not found. " + \
        f"After cloning benchmarks did you run setup.sh in {benchmark_repo}?"
    
    assert Path(vnnlib_filename).is_file(), f"vnnlib file not found: {vnnlib_filename}"

    ################################################

    res, msg = get_ce_diff(onnx_filename, vnnlib_filename, ce_path, Settings.COUNTEREXAMPLE_ATOL, Settings.COUNTEREXAMPLE_RTOL)

    print(f"CE result {res}: {msg}")
    
    return res

# @cachier(cache_dir='./cachier', stale_after=datetime.timedelta(days=7))
def get_ce_diff(onnx_filename, vnnlib_filename, ce_path, abs_tol, rel_tol):
    """get difference in execution"""

    try:
        content = read_ce_file(ce_path)
    except FileNotFoundError:
        return CounterexampleResult.NO_CE, f"Note: no counter example provided in {ce_path}"

    if len(content) < 2:
        return CounterexampleResult.NO_CE, f"Note: no counter example provided in {ce_path}"

    #print(f"CE CONTENT:\n{content}")
    
    assert content[0] == '(' and content[-1] == ')'
    content = content[1:-1]

    x_list = []
    y_list = []

    parts = content.split(')')
    for part in parts:
        part = part.strip()
                
        if not part:
            continue

        while "  " in part:
            part = part.replace("  ", " ")
        
        assert part[0] == '('
        part = part[1:]

        #print(f"part with len={len(part.split(' '))}: {part}")
        name, num = part.split(' ')
        assert name[0:2] in ['X_', 'Y_']

        if name[0:2] == 'X_':
            assert int(name[2:]) == len(x_list)
            x_list.append(float(num))
        else:
            assert int(name[2:]) == len(y_list)
            y_list.append(float(num))

    onnx_model = onnx.load(onnx_filename) 

    inp, _out, input_dtype = get_io_nodes(onnx_model)
    input_shape = tuple(d.dim_value if d.dim_value != 0 else 1 for d in inp.type.tensor_type.shape.dim)

    x_in = np.array(x_list, dtype=input_dtype)
    flatten_order = 'C'
    x_in = x_in.reshape(input_shape, order=flatten_order)
    output = predict_with_onnxruntime(onnx_model, x_in)

    flat_out = output.flatten(flatten_order)

    expected_y = np.array(y_list)
    extra_msg = ""

    if Settings.IGNORE_CE_Y:
        rel_error = 0
        msg = "Y from CE file ignored. Use onnxruntime prediction of Y instead."
        used_output = flat_out
    else:
        try:
            diff = np.linalg.norm(flat_out - expected_y, ord=np.inf)
            norm = np.linalg.norm(expected_y, ord=np.inf)
            if norm < 1e-6: # don't divide by zero
                rel_error = 0
            else:
                rel_error = diff / norm
        except ValueError as e:
            diff = 9999
            rel_error = 9999
            extra_msg = f" ERROR: {e}"
        msg = f"L-inf norm difference between onnx execution and CE file output: {diff} (rel error: {rel_error});"
        msg += f"(rel_limit: {rel_tol})"

        used_output = y_list

    #return diff, tuple(x_list), tuple(y_list)

    #diff, x_tup, y_tup = res

    msg += extra_msg
    rv = CounterexampleResult.CORRECT

    if rel_error > rel_tol:
        rv = CounterexampleResult.EXEC_DOESNT_MATCH
    else:
        # output matched onnxruntime, also need to check that the spec file was obeyed
        is_vio, msg2 = is_specification_vio(onnx_filename, vnnlib_filename, tuple(x_list), tuple(used_output), abs_tol)

        msg += "\n" + msg2

        if is_vio:
            # If the example is only valid because it's within the defined error tolerance,
            # this tool will not receive a penalty, but other tools may still correctly
            # prove UNSAT
            is_vio_zero_tolerance, _ = is_specification_vio(onnx_filename, vnnlib_filename, tuple(x_list), tuple(used_output), 0.0)
            if rel_error > 0 or not is_vio_zero_tolerance:
                msg += "\nNote: counterexample is not within bounds, but within error tolerance and will be accepted"
            rv = CounterexampleResult.CORRECT_UP_TO_TOLERANCE
        else:
            msg += "\nNote: counterexample in file did not violate the specification and so was invalid!"
            rv = CounterexampleResult.SPEC_NOT_VIOLATED

    return rv, msg

@cachier(cache_dir='./cachier', stale_after=datetime.timedelta(days=365), wait_for_calc_timeout=30, pickle_reload=False, separate_files=True)
def is_specification_vio(onnx_filename, vnnlib_filename, x_list, expected_y, tol):
    """check that the spec file was obeyed"""

    msg = "Checking if spec was actually violated"
    onnx_model = onnx.load(onnx_filename) 

    inp, out, _ = get_io_nodes(onnx_model)

    inp_shape = tuple(d.dim_value if d.dim_value != 0 else 1 for d in inp.type.tensor_type.shape.dim)
    out_shape = tuple(d.dim_value if d.dim_value != 0 else 1 for d in out.type.tensor_type.shape.dim)

    num_inputs = 1
    num_outputs = 1

    for n in inp_shape:
        num_inputs *= n

    for n in out_shape:
        num_outputs *= n

    box_spec_list = read_vnnlib_simple(vnnlib_filename, num_inputs, num_outputs)

    rv = False

    for i, box_spec in enumerate(box_spec_list):
        input_box, spec_list = box_spec
        assert len(input_box) == len(x_list), f"input box len: {len(input_box)}, x_in len: {len(x_list)}"

        inside_input_box = True

        for (lb, ub), x in zip(input_box, x_list):
            if x < lb - tol or x > ub + tol:
                inside_input_box = False
                break

        if inside_input_box:
            msg += f"\nCE input X was inside box #{i}"
            
            # check spec
            violated = False
                
            for j, (prop_mat, prop_rhs) in enumerate(spec_list):
                vec = prop_mat.dot(expected_y)
                sat = np.all(vec <= prop_rhs + tol)

                if sat:
                    msg += f"\nprop #{j} violated:\n{vec - prop_rhs}"
                    violated = True
                    break

            if violated:
                rv = True
                break
                
    return rv, msg

def test():
    """test code"""

    ce_filename = "test_ce.txt"
    cat = "cifar100_tinyimagenet_resnet"
    net = "TinyImageNet_resnet_medium"
    prop = "TinyImageNet_resnet_medium_prop_idx_6461_sidx_2771_eps_0.0039"

    #ce_filename = "mnist-net_256x2_prop_1_0.03.counterexample.gz"
    #net = "mnist-net_256x2"
    #prop = "prop_1_0.03"
    #cat = "mnist_fc"
    
    res = is_correct_counterexample(ce_filename, cat, net, prop)

    if res:
        print("counter example is correct")
    else:
        print("counter example is NOT correct")
        
if __name__ == "__main__":
    test()
