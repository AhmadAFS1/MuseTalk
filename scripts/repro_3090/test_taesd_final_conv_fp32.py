"""CPU-only tests. Real ONNX tests explicitly skip if ONNX/NumPy are absent."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import taesd_final_conv_fp32 as island

HAS_ONNX = importlib.util.find_spec('onnx') is not None and importlib.util.find_spec('numpy') is not None


def node(op, inputs, output, name=''):
    return {'op_type': op, 'input': list(inputs), 'output': [output], 'domain': '', 'name': name}


class TopologyTests(unittest.TestCase):
    def graph(self):
        return [node('Conv', ['x', 'w0'], 'a'), node('Relu', ['a'], 'b'),
                node('Conv', ['b', 'w1', 'b1'], 'c'), node('Clip', ['c'], 'y')]

    def select(self, nodes):
        return island.select_final_conv(nodes, ['x'], ['w0', 'w1', 'b1'], ['y'])

    def test_unique_final_conv_and_parameter_consumers(self):
        result = self.select(self.graph())
        self.assertEqual(result['target_index'], 2)
        self.assertEqual(result['parameter_names'], ['w1', 'b1'])
        self.assertEqual(result['output_reachable_conv_count'], 2)
        self.assertEqual(result['downstream_node_indices'], [3])

    def test_parallel_final_convs_fail_closed(self):
        nodes = [node('Conv', ['x', 'w0'], 'a'), node('Conv', ['x', 'w1'], 'b'), node('Add', ['a', 'b'], 'y')]
        with self.assertRaisesRegex(island.TransformRejected, 'unique_final'):
            self.select(nodes)

    def test_even_dead_downstream_conv_is_rejected(self):
        nodes = self.graph() + [node('Conv', ['c', 'w0'], 'unused')]
        with self.assertRaisesRegex(island.TransformRejected, 'downstream_conv'):
            self.select(nodes)

    def test_shared_target_weight_is_rejected(self):
        nodes = self.graph() + [node('Identity', ['w1'], 'unused')]
        with self.assertRaisesRegex(island.TransformRejected, 'shared_target'):
            self.select(nodes)

    def test_unknown_input_non_ssa_custom_domain_rejected(self):
        variants = []
        nodes = self.graph(); nodes[2]['input'][0] = 'unknown'; variants.append(nodes)
        nodes = self.graph(); nodes[2]['output'][0] = 'a'; variants.append(nodes)
        nodes = self.graph(); nodes[2]['domain'] = 'custom'; variants.append(nodes)
        for graph in variants:
            with self.assertRaises(island.TransformRejected):
                self.select(graph)

    def test_wrong_hash_rejected_before_imports_or_mutation(self):
        source = b'synthetic-source-not-an-onnx-model'
        with self.assertRaisesRegex(island.TransformRejected, 'source_sha256_mismatch'):
            island.transform(source, '0' * 64)
        self.assertEqual(source, b'synthetic-source-not-an-onnx-model')

    def test_plan_requires_hash_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source.onnx'
            source.write_bytes(b'plan only does not parse source')
            output, receipt = root / 'candidate.onnx', root / 'receipt.json'
            args = ['--input', str(source), '--expected-source-sha256', island.sha(source.read_bytes()),
                    '--output', str(output), '--receipt', str(receipt)]
            capture = io.StringIO()
            with patch.object(island, 'transform') as transform, contextlib.redirect_stdout(capture):
                self.assertEqual(island.main(args), 0)
            transform.assert_not_called()
            self.assertEqual(json.loads(capture.getvalue())['mode'], 'plan_only')
            self.assertFalse(output.exists() or receipt.exists())


@unittest.skipUnless(HAS_ONNX, 'Actual ONNX + NumPy packages not installed; graph/checker/inference tests not executed')
class RealOnnxTests(unittest.TestCase):
    def model(self):
        import numpy as np
        import onnx
        from onnx import helper as h, numpy_helper as n, TensorProto as t
        self.np, self.onnx = np, onnx
        initializers = [n.from_array(np.ones((3, 4, 1, 1), dtype=np.float16) * .125, 'w0'),
                        n.from_array(np.array([1, 1, 8, 8], dtype=np.float32), 'scales'),
                        n.from_array(np.arange(9, dtype=np.float16).reshape(3, 3, 1, 1) / 16, 'w1'),
                        n.from_array(np.array([-.0, .125, -.25], dtype=np.float16), 'b1'),
                        n.from_array(np.array(2, dtype=np.float16), 'two'),
                        n.from_array(np.array(.5, dtype=np.float16), 'half'),
                        n.from_array(np.array(0, dtype=np.float16), 'zero'),
                        n.from_array(np.array(1, dtype=np.float16), 'one')]
        nodes = [h.make_node('Conv', ['x', 'w0'], ['a'], name='first'),
                 h.make_node('Resize', ['a', '', 'scales'], ['up'], mode='nearest', coordinate_transformation_mode='asymmetric'),
                 h.make_node('Conv', ['up', 'w1', 'b1'], ['decoded'], name='actual_final'),
                 h.make_node('Div', ['decoded', 'two'], ['div']),
                 h.make_node('Add', ['div', 'half'], ['norm']),
                 h.make_node('Clip', ['norm', 'zero', 'one'], ['y'])]
        graph = h.make_graph(nodes, 'synthetic_taesd_shape_contract',
                             [h.make_tensor_value_info('x', t.FLOAT16, [8, 4, 32, 32])],
                             [h.make_tensor_value_info('y', t.FLOAT16, [8, 3, 256, 256])], initializers)
        result = h.make_model(graph, opset_imports=[h.make_opsetid('', 17)])
        result.producer_name = 'synthetic-only-no-GPU'
        return result

    def run_model(self, model):
        raw = island.wire(model)
        return island.transform(raw, island.sha(raw))

    def test_actual_checker_inference_casts_weights_and_immutable_source(self):
        model = self.model()
        before = island.wire(model)
        output, receipt = self.run_model(model)
        transformed = self.onnx.load_model_from_string(output)
        self.onnx.checker.check_model(transformed, full_check=True)
        self.assertEqual(island.wire(model), before)
        self.assertEqual(receipt['source_sha256'], island.sha(before))
        self.assertEqual(receipt['transformed_sha256'], island.sha(output))
        self.assertEqual(receipt['target']['target_index'], 2)
        self.assertEqual(receipt['mutation']['added_cast_nodes'], 2)
        self.assertTrue(receipt['mutation']['all_other_protobuf_fields_unchanged'])
        self.assertFalse(receipt['tensorrt_precision_verified'] or receipt['quality_accepted'] or receipt['release_ready'])
        original = {v.name: v for v in model.graph.initializer}
        for value in transformed.graph.initializer:
            if value.name in ('w1', 'b1'):
                self.assertEqual(value.data_type, self.onnx.TensorProto.FLOAT)
                self.assertEqual(self.onnx.numpy_helper.to_array(value).astype('<f2').tobytes(),
                                 self.onnx.numpy_helper.to_array(original[value.name]).astype('<f2').tobytes())
            else:
                self.assertEqual(island.wire(value), island.wire(original[value.name]))
        self.assertEqual(transformed.graph.node[2].op_type, 'Cast')
        self.assertEqual(transformed.graph.node[4].op_type, 'Cast')

    def test_reserved_collision_rejected(self):
        model = self.model()
        model.graph.node[0].name = island.PREFIX + 'cast_in'
        with self.assertRaisesRegex(island.TransformRejected, 'reserved_name_collision'):
            self.run_model(model)

    def test_shared_parameter_and_external_data_rejected(self):
        model = self.model()
        model.graph.node.append(self.onnx.helper.make_node('Identity', ['w1'], ['unused']))
        with self.assertRaisesRegex(island.TransformRejected, 'shared_target'):
            self.run_model(model)
        model = self.model()
        model.graph.initializer[0].data_location = self.onnx.TensorProto.EXTERNAL
        entry = model.graph.initializer[0].external_data.add(); entry.key = 'location'; entry.value = '/never/read'
        with self.assertRaisesRegex(island.TransformRejected, 'external_tensor'):
            self.run_model(model)

    def test_nonfinite_parameter_rejected(self):
        model = self.model()
        for value in model.graph.initializer:
            if value.name == 'b1':
                value.CopyFrom(self.onnx.numpy_helper.from_array(self.np.array([0, self.np.inf, 0], dtype=self.np.float16), 'b1'))
        with self.assertRaisesRegex(island.TransformRejected, 'nonfinite_target'):
            self.run_model(model)

    def test_wrong_boundary_and_symbolic_shape_rejected(self):
        model = self.model()
        model.graph.input[0].type.tensor_type.shape.dim[0].ClearField('dim_value')
        model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = 'unknown_batch'
        with self.assertRaises(island.TransformRejected):
            self.run_model(model)

    def test_deterministic_transform_and_nonoverwrite_cli(self):
        model = self.model()
        first, receipt1 = self.run_model(model)
        second, receipt2 = self.run_model(model)
        self.assertEqual(first, second)
        self.assertEqual(receipt1, receipt2)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / 'source.onnx'; source.write_bytes(island.wire(model))
            args = ['--input', str(source), '--expected-source-sha256', island.sha(source.read_bytes()),
                    '--output', str(root / 'new.onnx'), '--receipt', str(root / 'new.json'), '--execute']
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(island.main(args), 0)
            with self.assertRaisesRegex(island.TransformRejected, 'refusing_to_overwrite'):
                island.main(args)

    def test_structural_proof_catches_unrelated_constant_tamper(self):
        model = self.model()
        original_proof = island.prove_only_allowed_mutations
        def tamper(source, candidate, *args):
            candidate.producer_name = 'unexpected-metadata-mutation'
            return original_proof(source, candidate, *args)
        with patch.object(island, 'prove_only_allowed_mutations', side_effect=tamper):
            with self.assertRaisesRegex(island.TransformRejected, 'non_allowlisted'):
                self.run_model(model)


if __name__ == '__main__':
    unittest.main()
