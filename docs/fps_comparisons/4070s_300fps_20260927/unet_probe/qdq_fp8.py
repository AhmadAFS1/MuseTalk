"""Convert an INT8 Q/DQ ONNX graph (modelopt, max-calibrated) into an FP8-e4m3 Q/DQ graph (opset 19 standard ops).
scale_fp8 = scale_int8 * 127 / 448 (same amax). Optionally collapse per-channel weight scales to per-tensor."""
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto


def _const_value(model, name, const_nodes, inits):
    if name in inits:
        return numpy_helper.to_array(inits[name])
    if name in const_nodes:
        n = const_nodes[name]
        for a in n.attribute:
            if a.name == "value":
                return numpy_helper.to_array(a.t)
    raise KeyError(name)


def int8_qdq_to_fp8(onnx_bytes, per_tensor_weights=False):
    model = onnx.load_from_string(onnx_bytes)
    g = model.graph
    inits = {i.name: i for i in g.initializer}
    const_nodes = {n.output[0]: n for n in g.node if n.op_type == "Constant"}
    cnt = 0
    for node in g.node:
        if node.op_type not in ("QuantizeLinear", "DequantizeLinear"):
            continue
        sc_name = node.input[1]
        scale = _const_value(model, sc_name, const_nodes, inits).astype(np.float32)
        new_scale = scale * (127.0 / 448.0)
        axis_attr = [a for a in node.attribute if a.name == "axis"]
        if per_tensor_weights and new_scale.size > 1:
            new_scale = np.array(new_scale.max(), dtype=np.float32)
            for a in axis_attr:
                node.attribute.remove(a)
        # keep the scale's original dtype (fp16 or fp32 graphs)
        orig_dtype = _const_value(model, sc_name, const_nodes, inits).dtype
        sname = f"fp8_scale_{cnt}"
        zname = f"fp8_zp_{cnt}"
        cnt += 1
        g.initializer.append(numpy_helper.from_array(new_scale.astype(orig_dtype), sname))
        zp = helper.make_tensor(zname, TensorProto.FLOAT8E4M3FN, list(new_scale.shape),
                                vals=np.zeros(new_scale.size, dtype=np.uint8).tobytes(), raw=True)
        g.initializer.append(zp)
        node.input[1] = sname
        if len(node.input) > 2:
            node.input[2] = zname
        else:
            node.input.append(zname)
    for op in model.opset_import:
        if op.domain in ("", "ai.onnx"):
            op.version = max(op.version, 19)
    # drop stale value_info (int8 typed) so the parser re-infers
    del g.value_info[:]
    return model.SerializeToString(), cnt
