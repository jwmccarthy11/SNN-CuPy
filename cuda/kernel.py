import cupy as cp


def lif_update(params):
    return cp.ElementwiseKernel(
        in_params='float32 input, float32 v_in, int8 t_in',
        out_params='bool output, float32 v_out, int8 t_out',
        operation='''
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
        }}'''.format_map(params),
        name='lif_update'
    )


def propagate_delayed():
    return cp.RawKernel(
        code='''
        extern "C" __global__
        void propagate(
            bool* input, int t, int n_syn, int n_post, int d_max,
            float* output, float* w, int* d, int* i_pre, int* i_post
        ) {{
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n_syn) {{
                int x = i_post[idx];
                int y = (t + d[idx]) % d_max;
                if (input[i_pre[idx]]) {{
                    int out_idx = n_post * y + x;
                    atomicAdd(&output[out_idx], w[idx]);
                }}
            }}
        }}
        ''',
        name='propagate'
    )
