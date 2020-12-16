import cupy as cp


def lif_update(params):
    return cp.ElementwiseKernel(
        in_params="float32 input, float32 v_in, int8 t_in",
        out_params="bool output, float32 v_out, int8 t_out",
        operation="""
        if (t_in > 0) {{
            t_out = t_in - 1;
            v_out = v_in;
            output = 0;
        }} else {{
            v_out = v_in + (input * {rm} - v_in) / ({rm} * {cm});
            if (v_out >= {thr}) {{
                v_out = {res};
                t_out = {ref};
                output = 1;
            }} else {{
                output = 0;
            }}
        }}""".format_map(params),
        name="lif_update"
    )

def full_propagate():
    pass

def sparse_propagate():
    pass