"""Exercise the backported registry with two distinct Adafactor classes."""
import ast
import inspect
from types import SimpleNamespace
import unittest

from patch_mmengine_compat import patched_source

SOURCE = '''def register_torch_optimizers():
    torch_optimizers = []
    for module_name in dir(torch.optim):
        if module_name.startswith('__'):
            continue
        _optim = getattr(torch.optim, module_name)
        if inspect.isclass(_optim) and issubclass(_optim,
                                                  torch.optim.Optimizer):
            OPTIMIZERS.register_module(module=_optim)
            torch_optimizers.append(module_name)
    return torch_optimizers
'''


class PatchTests(unittest.TestCase):
    def test_torch_and_transformers_optimizers_keep_distinct_names(self):
        class Optimizer: pass
        class Adafactor(Optimizer): pass
        class SGD(Optimizer): pass
        transformer_adafactor = type("Adafactor", (Optimizer,), {})
        registry = {}
        def register_module(module, name=None):
            name = name or module.__name__
            if name in registry:
                raise KeyError(name)
            registry[name] = module
        namespace = {"torch": SimpleNamespace(optim=SimpleNamespace(Optimizer=Optimizer, Adafactor=Adafactor, SGD=SGD)),
                     "inspect": inspect, "OPTIMIZERS": SimpleNamespace(register_module=register_module)}
        patched, changed = patched_source(SOURCE)
        self.assertTrue(changed)
        exec(compile(ast.parse(patched), "patched-mmengine", "exec"), namespace)
        namespace["register_torch_optimizers"]()
        register_module(transformer_adafactor, name="Adafactor")
        self.assertIs(registry["TorchAdafactor"], Adafactor)
        self.assertIs(registry["Adafactor"], transformer_adafactor)
        self.assertIs(registry["SGD"], SGD)

    def test_idempotent_and_other_functions_unchanged(self):
        source = SOURCE + '\ndef unrelated():\n    return "Adafactor"\n'
        patched, _ = patched_source(source)
        twice, changed = patched_source(patched)
        self.assertFalse(changed)
        self.assertEqual(twice, patched)
        self.assertTrue(patched.endswith('\ndef unrelated():\n    return "Adafactor"\n'))

    def test_unknown_source_is_rejected(self):
        for source in (SOURCE.replace("module=_optim", "module=other"),
                       SOURCE.replace("register_torch_optimizers", "different"), SOURCE + SOURCE):
            with self.assertRaises(ValueError):
                patched_source(source)


if __name__ == "__main__":
    unittest.main()
