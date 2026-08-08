import ast
import importlib.util
import unittest
from pathlib import Path


SOURCE_PATH = Path(__file__).parents[3] / "olmo_core/nn/mamba3/mamba3_ssd_fast.py"
ACCOUNTING_PATH = Path(__file__).parents[3] / "olmo_core/nn/mamba3/b3_speed_accounting.py"


def _method(class_name: str, method_name: str) -> ast.FunctionDef:
    module = ast.parse(SOURCE_PATH.read_text())
    cls = next(
        node for node in module.body if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )


def _function(function_name: str) -> ast.FunctionDef:
    module = ast.parse(SOURCE_PATH.read_text())
    return next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )


class B3SavedStateContractTest(unittest.TestCase):
    def test_fused_backward_saves_inputs_not_materialized_bc_vectors(self) -> None:
        forward = _method("_FusedQuaternionRotateBC", "forward")
        saves = [
            call
            for call in ast.walk(forward)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "save_for_backward"
        ]

        self.assertEqual(len(saves), 1)
        self.assertEqual(
            [arg.id for arg in saves[0].args if isinstance(arg, ast.Name)],
            ["B", "C", "theta", "prefix"],
        )

        backward = _method("_FusedQuaternionRotateBC", "backward")
        assigned_saved_names = [
            element.id
            for node in ast.walk(backward)
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "saved_tensors"
            for target in node.targets
            if isinstance(target, ast.Tuple)
            for element in target.elts
            if isinstance(element, ast.Name)
        ]
        self.assertEqual(assigned_saved_names, ["B", "C", "theta", "prefix"])

        backward_source = ast.unparse(backward)
        self.assertIn("torch.cat((B, C), dim=-2)", backward_source)
        self.assertNotIn("vectors = ctx.saved_tensors", backward_source)
        self.assertIn("_quaternion_prefix_backward(prefix, grad_prefix)", backward_source)
        self.assertNotIn("jacobian", backward_source)

        forward_source = ast.unparse(forward)
        self.assertNotIn("_quaternion_to_matrix", forward_source)
        self.assertNotIn("(3, 3)", forward_source)

    def test_production_shape_accounting_exposes_eliminated_saved_storage(self) -> None:
        self.assertTrue(ACCOUNTING_PATH.is_file(), "b=3 accounting helper is missing")
        spec = importlib.util.spec_from_file_location("b3_speed_accounting", ACCOUNTING_PATH)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        accounting = module.saved_tensor_accounting(
            B_shape=(8, 4096, 1, 1, 192),
            C_shape=(8, 4096, 1, 1, 192),
            theta_shape=(8, 4096, 1, 64, 3),
            bc_element_size=2,
        )

        self.assertEqual(accounting["materialized_bc_saved_bytes_avoided"], 24 * 1024 * 1024)
        self.assertEqual(accounting["compact_prefix_saved_bytes"], 32 * 1024 * 1024)
        self.assertEqual(accounting["matrix_prefix_bytes_avoided"], 72 * 1024 * 1024)
        self.assertEqual(accounting["generated_saved_bytes"], 32 * 1024 * 1024)

    def test_public_recurrence_signature_remains_checkpoint_compatible(self) -> None:
        function = _function("mamba3_ssd_fast")
        self.assertEqual(
            [argument.arg for argument in function.args.args],
            ["x", "B", "C", "dt", "A", "lam", "theta"],
        )
        self.assertEqual(
            [argument.arg for argument in function.args.kwonlyargs],
            [
                "heads_per_group",
                "block_size",
                "chunk_size",
                "rotation_scan_chunk",
                "rotation_scan_impl",
                "selective_fp32",
            ],
        )


if __name__ == "__main__":
    unittest.main()
