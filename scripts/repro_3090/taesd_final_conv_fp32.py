#!/usr/bin/env python3
"""Isolated CPU ONNX final-Conv FP32-island proposal; no builder/loader changes.

The exact pinned source graph stays immutable. Only the unique last
output-reachable Conv receives FP32 activation and exactly promoted FP16-rounded
weights/bias, then casts back to FP16. All other protobuf fields are proven
unchanged. This is NOT a TensorRT precision, quality, or performance acceptance:
a future reviewed builder must disable TF32 and enforce strongly typed execution.

CLI defaults to plan-only. --execute writes two new files, never overwrites.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import sys

RECIPE = 'taesd_final_conv_fp32_island_v1'
PREFIX = '__musetalk_taesd_final_conv_fp32_v1_'


class TransformRejected(ValueError):
    pass


def require(value, reason):
    if not value:
        raise TransformRejected(reason)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def wire(proto):
    return proto.SerializeToString(deterministic=True)


def select_final_conv(nodes, inputs, initializers, outputs):
    """Stdlib SSA/topology analysis; indices are stable source-node identities."""
    require(len(inputs) == len(outputs) == 1, 'single_input_output_required')
    initializers = list(initializers)
    require(len(set(initializers)) == len(initializers), 'duplicate_initializer')
    require(not set(inputs) & set(initializers), 'initializer_graph_input_alias')
    known = set(inputs) | set(initializers)
    producer = {}
    consumers = {}
    for index, node in enumerate(nodes):
        require(node.get('domain', '') in ('', 'ai.onnx'), 'custom_domain_forbidden')
        for name in node['input']:
            if name:
                require(name in known, 'unknown_input_or_non_topological_graph')
                consumers.setdefault(name, set()).add(index)
        for name in node['output']:
            require(bool(name) and name not in known, 'duplicate_or_empty_tensor_output')
            known.add(name)
            producer[name] = index
    require(all(name in producer for name in outputs), 'graph_output_not_produced')
    upstream = set()
    todo = [producer[name] for name in outputs]
    while todo:
        index = todo.pop()
        if index in upstream:
            continue
        upstream.add(index)
        todo += [producer[name] for name in nodes[index]['input'] if name in producer]

    def downstream(index):
        found = set()
        todo = [j for name in nodes[index]['output'] for j in consumers.get(name, ())]
        while todo:
            item = todo.pop()
            if item in found:
                continue
            found.add(item)
            todo += [j for name in nodes[item]['output'] for j in consumers.get(name, ())]
        return found

    convs = {i for i in upstream if nodes[i]['op_type'] == 'Conv'}
    terminal = [i for i in convs if not (downstream(i) & convs)]
    require(len(terminal) == 1, 'unique_final_output_reachable_conv_required')
    target = terminal[0]
    after = downstream(target)
    require(not any(nodes[i]['op_type'] == 'Conv' for i in after), 'downstream_conv_forbidden')
    node = nodes[target]
    require(len(node['input']) in (2, 3) and all(node['input']) and len(node['output']) == 1,
            'unsupported_conv_signature')
    parameters = node['input'][1:]
    require(len(parameters) == len(set(parameters)), 'aliased_conv_parameters')
    for name in parameters:
        require(name in initializers, 'embedded_initializer_required')
        require(consumers.get(name) == {target}, 'shared_target_initializer_forbidden')
        require(name not in outputs, 'target_parameter_is_graph_output')
    return {'target_index': target, 'parameter_names': parameters,
            'output_reachable_conv_count': len(convs), 'downstream_node_indices': sorted(after)}


def _types(model):
    result = {}
    for value in (*model.graph.input, *model.graph.output, *model.graph.value_info):
        tensor = value.type.tensor_type
        # Empty Resize ROI constants and unrelated symbolic annotations can be
        # valid ONNX. Only the six actual island/I/O boundaries must be fully
        # static and positive; their exact shapes are checked below.
        shape = ([dim.dim_value if dim.HasField('dim_value') else None
                  for dim in tensor.shape.dim] if tensor.HasField('shape') else None)
        current = (tensor.elem_type, shape)
        require(value.name not in result or result[value.name] == current, 'conflicting_tensor_annotations')
        result[value.name] = current
    return result


def _no_external_or_subgraphs(model, onnx):
    require(not model.functions and not model.training_info and not model.graph.sparse_initializer,
            'functions_training_or_sparse_initializers_forbidden')
    tensors = list(model.graph.initializer)
    for node in model.graph.node:
        for attribute in node.attribute:
            require(attribute.type not in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS),
                    'nested_graph_forbidden')
            if attribute.type == onnx.AttributeProto.TENSOR:
                tensors.append(attribute.t)
            elif attribute.type == onnx.AttributeProto.TENSORS:
                tensors.extend(attribute.tensors)
            require(attribute.type not in (onnx.AttributeProto.SPARSE_TENSOR, onnx.AttributeProto.SPARSE_TENSORS),
                    'sparse_attribute_forbidden')
    require(all(t.data_location != onnx.TensorProto.EXTERNAL and not t.external_data for t in tensors),
            'external_tensor_data_forbidden')


def prove_only_allowed_mutations(source, candidate, target, casts, promoted, new_values):
    """Undo the exact allowlisted delta, then require full protobuf identity."""
    restored = copy.deepcopy(candidate)
    index = target['target_index']
    original_node = source.graph.node[index]
    expected = copy.deepcopy(original_node)
    expected.input[0] = PREFIX + 'activation_fp32'
    expected.output[0] = PREFIX + 'conv_fp32'
    require(len(restored.graph.node) == len(source.graph.node) + 2, 'unexpected_node_count')
    require(wire(restored.graph.node[index]) == wire(casts[0]) and
            wire(restored.graph.node[index + 1]) == wire(expected) and
            wire(restored.graph.node[index + 2]) == wire(casts[1]), 'unexpected_target_or_cast_mutation')
    del restored.graph.node[index:index + 3]
    restored.graph.node.insert(index, copy.deepcopy(original_node))
    originals = {x.name: x for x in source.graph.initializer}
    require(len(restored.graph.initializer) == len(originals), 'unexpected_initializer_count')
    for initializer in restored.graph.initializer:
        if initializer.name in promoted:
            require(wire(initializer) == wire(promoted[initializer.name]), 'unexpected_promoted_parameter')
            initializer.CopyFrom(originals[initializer.name])
    require(len(restored.graph.value_info) == len(source.graph.value_info) + 2, 'unexpected_value_info_count')
    require([wire(v) for v in restored.graph.value_info[-2:]] == [wire(v) for v in new_values],
            'unexpected_boundary_annotations')
    del restored.graph.value_info[-2:]
    require(wire(restored) == wire(source), 'non_allowlisted_graph_or_metadata_mutation')


def transform(source_bytes, expected_source_sha256):
    require(isinstance(source_bytes, bytes), 'immutable_source_bytes_required')
    require(isinstance(expected_source_sha256, str) and re.fullmatch(r'[0-9a-f]{64}', expected_source_sha256),
            'explicit_source_sha256_required')
    require(sha(source_bytes) == expected_source_sha256, 'source_sha256_mismatch')
    try:
        import numpy as np
        import onnx
    except ImportError:
        raise TransformRejected('onnx_and_numpy_required_not_available') from None
    try:
        source = onnx.load_model_from_string(source_bytes)
        _no_external_or_subgraphs(source, onnx)  # before checker; never resolve external paths
        require([(x.domain, x.version) for x in source.opset_import] == [('', 17)], 'exact_opset17_required')
        onnx.checker.check_model(source, full_check=True)
        boundary = _types(source)
        require(len(source.graph.input) == len(source.graph.output) == 1, 'single_input_output_required')
        input_name, output_name = source.graph.input[0].name, source.graph.output[0].name
        require(boundary[input_name] == (onnx.TensorProto.FLOAT16, [8, 4, 32, 32]), 'wrong_source_input_boundary')
        require(boundary[output_name] == (onnx.TensorProto.FLOAT16, [8, 3, 256, 256]), 'wrong_source_output_boundary')
        nodes = [{'name': n.name, 'op_type': n.op_type, 'domain': n.domain,
                  'input': list(n.input), 'output': list(n.output)} for n in source.graph.node]
        target = select_final_conv(nodes, [input_name], [x.name for x in source.graph.initializer], [output_name])
        names = {name for n in nodes for name in (n['name'], *n['input'], *n['output'])}
        names.update(x.name for x in (*source.graph.initializer, *source.graph.input,
                                     *source.graph.output, *source.graph.value_info))
        require(not any(name.startswith(PREFIX) for name in names), 'reserved_name_collision')
        inferred = onnx.shape_inference.infer_shapes(source, check_type=True, strict_mode=True, data_prop=True)
        before_types = _types(inferred)
        original_node = source.graph.node[target['target_index']]
        activation, conv_output = original_node.input[0], original_node.output[0]
        require(activation in before_types and conv_output in before_types, 'target_shape_inference_incomplete')
        activation_dtype, activation_shape = before_types[activation]
        require(activation_dtype == onnx.TensorProto.FLOAT16 and activation_shape is not None and len(activation_shape) == 4
                and activation_shape[0] == 8 and activation_shape[2:] == [256, 256], 'wrong_final_conv_activation')
        require(isinstance(activation_shape[1], int) and activation_shape[1] > 0, 'unknown_activation_channels')
        require(before_types[conv_output] == (onnx.TensorProto.FLOAT16, [8, 3, 256, 256]), 'wrong_final_conv_output')
        group = [a.i for a in original_node.attribute if a.name == 'group']
        require(not group or group == [1], 'grouped_final_conv_forbidden')
        candidate = copy.deepcopy(source)
        promoted, parameter_receipts = {}, []
        for initializer in candidate.graph.initializer:
            if initializer.name not in target['parameter_names']:
                continue
            original_wire = wire(initializer)
            require(initializer.data_type == onnx.TensorProto.FLOAT16, 'target_parameter_not_fp16')
            array = onnx.numpy_helper.to_array(initializer)
            require(np.isfinite(array).all(), 'nonfinite_target_parameter')
            if initializer.name == original_node.input[1]:
                require(array.ndim == 4 and array.shape[:2] == (3, activation_shape[1]), 'wrong_conv_weight_shape')
            else:
                require(array.shape == (3,), 'wrong_conv_bias_shape')
            widened = array.astype('<f4')
            original_values = array.astype('<f2', copy=False).tobytes()
            require(widened.astype('<f2').tobytes() == original_values, 'fp16_weight_roundtrip_changed')
            for field in ('raw_data', 'float_data', 'int32_data', 'int64_data', 'double_data', 'uint64_data', 'string_data'):
                initializer.ClearField(field)
            initializer.data_type = onnx.TensorProto.FLOAT
            initializer.raw_data = widened.tobytes()
            promoted[initializer.name] = copy.deepcopy(initializer)
            parameter_receipts.append({'name': initializer.name, 'shape': list(array.shape),
                'source_dtype': 'FLOAT16', 'target_dtype': 'FLOAT', 'source_initializer_sha256': sha(original_wire),
                'source_fp16_value_bytes_sha256': sha(original_values),
                'promoted_fp32_value_bytes_sha256': sha(widened.tobytes()),
                'target_initializer_sha256': sha(wire(initializer)), 'fp16_roundtrip_bit_exact': True})
        require(len(promoted) == len(target['parameter_names']), 'missing_target_parameter')
        cast_in = onnx.helper.make_node('Cast', [activation], [PREFIX + 'activation_fp32'],
                                       name=PREFIX + 'cast_in', to=onnx.TensorProto.FLOAT)
        cast_out = onnx.helper.make_node('Cast', [PREFIX + 'conv_fp32'], [conv_output],
                                        name=PREFIX + 'cast_out', to=onnx.TensorProto.FLOAT16)
        index = target['target_index']
        candidate.graph.node[index].input[0] = PREFIX + 'activation_fp32'
        candidate.graph.node[index].output[0] = PREFIX + 'conv_fp32'
        candidate.graph.node.insert(index, cast_in)
        candidate.graph.node.insert(index + 2, cast_out)
        new_values = [onnx.helper.make_tensor_value_info(PREFIX + 'activation_fp32', onnx.TensorProto.FLOAT, activation_shape),
                      onnx.helper.make_tensor_value_info(PREFIX + 'conv_fp32', onnx.TensorProto.FLOAT, [8, 3, 256, 256])]
        candidate.graph.value_info.extend(new_values)
        prove_only_allowed_mutations(source, candidate, target, [cast_in, cast_out], promoted, new_values)
        onnx.checker.check_model(candidate, full_check=True)
        after_types = _types(onnx.shape_inference.infer_shapes(candidate, check_type=True, strict_mode=True, data_prop=True))
        for name, expected in ((input_name, (onnx.TensorProto.FLOAT16, [8, 4, 32, 32])),
                               (output_name, (onnx.TensorProto.FLOAT16, [8, 3, 256, 256])),
                               (activation, (onnx.TensorProto.FLOAT16, activation_shape)),
                               (PREFIX + 'activation_fp32', (onnx.TensorProto.FLOAT, activation_shape)),
                               (PREFIX + 'conv_fp32', (onnx.TensorProto.FLOAT, [8, 3, 256, 256])),
                               (conv_output, (onnx.TensorProto.FLOAT16, [8, 3, 256, 256]))):
            require(after_types.get(name) == expected, 'transformed_boundary_shape_or_dtype_not_proven')
        output = wire(candidate)
        require(sha(source_bytes) == expected_source_sha256, 'immutable_source_changed')
        receipt = {'schema': 'taesd_final_conv_fp32_mutation_v1', 'recipe': RECIPE,
            'status': 'GRAPH_TRANSFORM_VERIFIED_NOT_ENGINE_OR_QUALITY_ACCEPTANCE',
            'source_sha256': expected_source_sha256, 'source_parsed_proto_sha256': sha(wire(source)),
            'transformed_sha256': sha(output), 'source_bytes': len(source_bytes), 'transformed_bytes': len(output),
            'onnx_version': onnx.__version__, 'numpy_version': np.__version__,
            'target': {**target, 'node_name': original_node.name, 'source_node_sha256': sha(wire(original_node)),
                       'original_input_names': list(original_node.input), 'original_output_names': list(original_node.output)},
            'parameters': parameter_receipts, 'mutation': {'added_cast_nodes': 2, 'rewired_conv_nodes': 1,
                'promoted_initializers': len(promoted), 'added_value_annotations': 2,
                'all_other_protobuf_fields_unchanged': True, 'existing_constants_other_than_target_parameters_unchanged': True},
            'proof': {'source_checker': True, 'candidate_checker': True, 'strict_shape_inference': True,
                'fp16_input_shape': [8, 4, 32, 32], 'fp16_output_shape': [8, 3, 256, 256],
                'exactly_one_final_conv': True, 'no_downstream_conv': True},
            'weights_identity_scope': 'embedded already-rounded FP16 ONNX values; original checkpoint not independently hashed here',
            'required_future_builder': {'strongly_typed': True, 'tf32_disabled': True, 'batch': 8,
                'optimization_level': 3, 'hardware_compatibility': 'none', 'fresh_engine_directory_and_fingerprint': True},
            'tensorrt_precision_verified': False, 'quality_accepted': False, 'performance_measured': False,
            'default_selection_changed': False, 'release_ready': False}
        return output, receipt
    except TransformRejected:
        raise
    except Exception as exc:
        # Do not expose arbitrary model contents/paths from parser diagnostics.
        raise TransformRejected('onnx_validation_failed_' + type(exc).__name__) from None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--expected-source-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    require(args.input.is_file() and not args.input.is_symlink(), 'regular_input_required')
    require(len({p.resolve() for p in (args.input, args.output, args.receipt)}) == 3, 'distinct_paths_required')
    require(not args.output.exists() and not args.output.is_symlink()
            and not args.receipt.exists() and not args.receipt.is_symlink(), 'refusing_to_overwrite')
    source = args.input.read_bytes()
    require(re.fullmatch(r'[0-9a-f]{64}', args.expected_source_sha256)
            and sha(source) == args.expected_source_sha256, 'source_sha256_mismatch')
    if not args.execute:
        print(json.dumps({'mode': 'plan_only', 'source_sha256': sha(source), 'recipe': RECIPE,
                          'gpu_actions': False, 'files_written': False, 'quality_accepted': False}))
        return 0
    output, receipt = transform(source, args.expected_source_sha256)
    # Both paths must be new. A partial write is never overwritten/reused on retry.
    with args.output.open('xb') as handle:
        handle.write(output)
    with args.receipt.open('x') as handle:
        json.dump(receipt, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'status': receipt['status'], 'source_sha256': receipt['source_sha256'],
                      'transformed_sha256': receipt['transformed_sha256'], 'release_ready': False}))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except TransformRejected as exc:
        print(json.dumps({'status': 'REJECTED', 'reason': str(exc), 'release_ready': False}))
        raise SystemExit(2)
