"""Numerical and hook tests; never downloads or loads pretrained weights."""
from __future__ import annotations

import copy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from pilot.config import DEFAULTS, load
from pilot.data import coverage, make, read, split_for
from pilot.preflight import fixture, gradient_error
from pilot.reporting import paired_bootstrap
from pilot.steering import steer
from pilot.storage import manifest


class NumericalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.probe, cls.geometry, cls.mu, cls.basis, cls.validation = fixture(features=256)
        cls.settings = copy.deepcopy(DEFAULTS["steering"])

    def test_analytic_gradient_against_finite_difference(self):
        error, radial = gradient_error(self.probe, self.validation[0])
        self.assertLess(error, 1e-5)
        self.assertLess(abs(radial), 1e-8)

    def test_prior_correction_changes_threshold_not_direction(self):
        x = self.validation[:5]
        shift = self.probe.log_dre(x)-self.probe.logits(x)
        np.testing.assert_allclose(shift, np.log((1-self.probe.prior)/self.probe.prior))

    def test_zero_is_identity_except_diagnostic(self):
        for method in DEFAULTS["steering"]["methods"] + ["E3"]:
            if method == "R0":
                continue
            source = self.mu+15*self.basis[:, 8]  # Includes rejected inputs.
            result = steer(source, method, 0, self.probe, self.geometry, self.settings)
            np.testing.assert_array_equal(result.states[-1], source)

    def test_small_strength_bound_even_without_entry_gate(self):
        for source in (self.validation[0], self.mu+.1*self.basis[:, 0]):
            for method in ("E2", "C-end", "E3"):
                for t in (1e-6, 1e-4, .01):
                    result = steer(source, method, t, self.probe, self.geometry, self.settings, gate=False)
                    self.assertLessEqual(result.displacement, 1.25*t+1e-8)

    def test_projection_preserves_residual(self):
        source = self.validation[0]
        for method in ("C", "C-end", "E2", "E3", "F"):
            result = steer(source, method, 3, self.probe, self.geometry, self.settings, gate=False)
            delta = result.states[-1]-source
            np.testing.assert_allclose(delta, (delta @ self.geometry.u) @ self.geometry.u.T, atol=1e-8)

    def test_spherical_update_preserves_norm(self):
        source = self.validation[0]
        result = steer(source, "S", 8, self.probe, self.geometry, self.settings, gate=False)
        np.testing.assert_allclose(np.linalg.norm(result.states, axis=1), np.linalg.norm(source), atol=1e-8)

    def test_unsupported_start_is_unchanged(self):
        source = self.mu+100*self.basis[:, 8]
        result = steer(source, "E2", 8, self.probe, self.geometry, self.settings)
        self.assertFalse(result.eligible)
        self.assertEqual(result.reason, "rejected_start")
        np.testing.assert_array_equal(result.states[-1], source)

    def test_probe_round_trip(self):
        from pilot.probe import Probe
        restored = Probe.restore(self.probe.arrays())
        np.testing.assert_allclose(restored.log_dre(self.validation[:3]), self.probe.log_dre(self.validation[:3]))
        np.testing.assert_allclose(restored.grad(self.validation[:3]), self.probe.grad(self.validation[:3]))

    def test_baselines_move_rejected_starts(self):
        source = self.mu+100*self.basis[:, 8]
        for method in ("A", "B", "S"):
            result = steer(source, method, 2, self.probe, self.geometry, self.settings)
            self.assertFalse(result.eligible)
            self.assertEqual(result.reason, "ok")
            self.assertGreater(result.displacement, 0)

    def test_support_exit_pads_remaining_solver_time(self):
        with patch.object(self.geometry, "eligible", side_effect=[np.array([True]), np.array([True]), np.array([False])]):
            result = steer(self.validation[0], "E2", 2, self.probe, self.geometry, self.settings)
        self.assertEqual(result.reason, "support_exit_frozen")
        self.assertGreater(np.linalg.norm(result.states[1]-result.states[0]), 0)
        np.testing.assert_array_equal(result.states[1:], np.repeat(result.states[1:2], 10, axis=0))

    def test_endpoint_correction_keeps_previous_checkpoint_times(self):
        source = self.validation[0]
        plain = steer(source, "C", 2, self.probe, self.geometry, self.settings, gate=False)
        corrected = steer(source, "C-end", 2, self.probe, self.geometry, self.settings, gate=False)
        np.testing.assert_array_equal(plain.states[:-1], corrected.states[:-1])


class DataAndAnalysisTests(unittest.TestCase):
    def test_supplied_profiles_cover_bands_and_fit_decode_budgets(self):
        from pilot.pipeline import estimate_cost
        config_dir = Path(__file__).resolve().parents[1] / "configs"
        for profile in ("cpu-smoke.json", "mps-128gb.json"):
            with self.subTest(profile=profile), tempfile.TemporaryDirectory() as folder:
                cfg = load(config_dir / profile)
                cfg["output"] = folder
                summary = make(cfg)
                self.assertTrue(summary["coverage"]["adequate"], summary["coverage"])
                for split, paths in (("validation", False), ("validation", True), ("test", False)):
                    estimate = estimate_cost(cfg, split, paths=paths)
                    self.assertLessEqual(estimate["decode_examples_upper_bound"], estimate["budget"])

    def test_split_identity_across_templates_and_hex_deduplication(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg = copy.deepcopy(DEFAULTS)
            cfg["output"] = folder
            cfg["data"].update(hue_step=4, saturations=[.8], lightnesses=[.5], templates=["main", "paraphrase"])
            make(cfg)
            rows = read(cfg)
            self.assertEqual(len({r["id"] for r in rows}), len(rows))
            groups = {}
            for row in rows:
                groups.setdefault(row["hex"], set()).add(row["split"])
            self.assertTrue(all(len(s) == 1 for s in groups.values()))
            self.assertTrue(all(r["split"] in {"validation", "test"} for r in rows if r["kind"] == "unrelated"))

    def test_manifest_rejects_mixed_settings(self):
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder)/"manifest.json"
            manifest(file, {"value": 1})
            with self.assertRaises(ValueError):
                manifest(file, {"value": 2})

    def test_source_only_changes_do_not_invalidate_artifacts(self):
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder)/"dataset-manifest.json"
            original = manifest(file, {"value": 1})
            with patch.object(Path, "read_bytes", return_value=b"# comment-only edit"):
                self.assertEqual(manifest(file, {"value": 1}), original)

    def test_coverage_does_not_count_paraphrases_as_new_examples(self):
        cfg = copy.deepcopy(DEFAULTS)
        rows = [{"id": template, "hex": "#00FF00", "hue": 150, "primary": True,
                 "kind": "colour", "split": "test", "template": template} for template in ("main", "paraphrase")]
        result = coverage(rows, cfg)
        self.assertEqual(result["primary_unique_hexes"], 1)
        self.assertEqual(result["splits"]["test"]["150"]["count"], 1)
        self.assertFalse(result["adequate"])

    def test_bootstrap_pairs_prompts_and_keeps_seed_summary(self):
        left = [{"id": str(i), "seed": seed, "loss": 100*i+seed} for i in range(5) for seed in range(3)]
        right = [{**r, "loss": r["loss"]+2} for r in reversed(left)]
        result = paired_bootstrap(left, right, "loss", 500, 8)
        self.assertEqual(result["difference"], -2)
        self.assertEqual(result["interval"], [-2, -2])
        self.assertEqual(result["prompts"], 5)
        self.assertEqual(result["seeds"], 3)


class HookTests(unittest.TestCase):
    def _runtime(self, device="cpu", tuple_output=False):
        import torch
        from types import SimpleNamespace
        from pilot.model import Runtime

        class Tokenizer:
            def __call__(self, texts, **kwargs):
                lengths = [min(7, max(2, len(t.split()))) for t in texts]
                width = max(lengths)
                ids = torch.zeros((len(texts), width), dtype=torch.long)
                mask = torch.zeros_like(ids)
                for i, length in enumerate(lengths):
                    ids[i, -length:] = torch.arange(1, length+1)
                    mask[i, -length:] = 1
                return {"input_ids": ids, "attention_mask": mask}

        class Layer(torch.nn.Module):
            def forward(self, x):
                out = x+.25
                return (out, "preserved") if tuple_output else out

        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(10, 8)
                self.layers = torch.nn.ModuleList([Layer()])
                self.head = torch.nn.Linear(8, 10, bias=False)

            def forward(self, input_ids, **kwargs):
                output = self.layers[0](self.embedding(input_ids))
                hidden = output[0] if isinstance(output, tuple) else output
                return SimpleNamespace(logits=self.head(hidden))

        torch.manual_seed(5)
        r = Runtime.__new__(Runtime)
        r.torch, r.device = torch, torch.device(device)
        r.cfg = copy.deepcopy(DEFAULTS)
        r.model = Tiny().to(device).eval()
        r.layers, r.tokenizer = r.model.layers, Tokenizer()
        r.token_ids = [[i] for i in range(8)]
        r.last_logits_only = False
        return r

    def test_identity_patch_and_cleanup_for_both_output_types(self):
        rows = [{"kind": "unrelated", "text": t} for t in ("short prompt", "a longer prompt to pad")]
        for tuple_output in (False, True):
            runtime = self._runtime(tuple_output=tuple_output)
            p, a, _, _ = runtime.forward(rows, capture=[0])
            patched, _, _, _ = runtime.forward(rows, patch_layer=0, replacements=a[0])
            np.testing.assert_allclose(p, patched, atol=1e-7)
            np.testing.assert_allclose(p.sum(1), 1, atol=1e-7)
            self.assertFalse(runtime.layers[0]._forward_hooks)
            with self.assertRaises(ValueError):
                runtime.forward(rows, patch_layer=0, replacements=np.zeros((2, 3)))
            self.assertFalse(runtime.layers[0]._forward_hooks)

    def test_decode_cache_reuses_identical_states_and_retains_full_vocabulary(self):
        from pilot.pipeline import decode_states
        runtime = self._runtime()
        runtime.metadata = {"device": "cpu"}
        row = {"id": "control", "kind": "unrelated", "text": "a sample prompt"}
        _, captured, _, _ = runtime.forward([row], capture=[0])
        states = np.repeat(captured[0], 3, axis=0)
        with tempfile.TemporaryDirectory() as folder:
            cfg = copy.deepcopy(DEFAULTS)
            cfg["output"] = folder
            first, full, _, count = decode_states(cfg, runtime, {"model": "mock"}, row, 0, states)
            self.assertEqual(count, 1)
            self.assertEqual(full.shape, (3, 10))
            with patch.object(runtime, "batched", side_effect=AssertionError("Cache should avoid forwarding")):
                again, _, _, count = decode_states(cfg, runtime, {"model": "mock"}, row, 0, states)
            self.assertEqual(count, 0)
            np.testing.assert_array_equal(first, again)

    @unittest.skipUnless(os.environ.get("PILOT_TEST_ACCELERATORS") == "1", "Accelerator tests require explicit opt-in")
    def test_accelerator_hook_matches_cpu(self):
        import torch
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else None
        if device is None:
            self.skipTest("No accelerator available")
        rows = [{"kind": "unrelated", "text": "a sample prompt"}]
        cpu, gpu = self._runtime(), self._runtime(device=device)
        a, _, _, _ = cpu.forward(rows, capture=[0])
        b, _, _, _ = gpu.forward(rows, capture=[0])
        np.testing.assert_allclose(a, b, atol=1e-5, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
