"""
Unit and Integration Tests for OPM-Synth Studio.
"""

import unittest
from pathlib import Path
import numpy as np

from engine.deck_io import DeckParser
from engine.geostats import GeostatisticalEngine
from engine.petrophysics import PetrophysicsEngine
from engine.proxy_flow import ReservoirProxySimulator
from engine.validator import ReservoirValidator
from generator import SyntheticDataGenerator


class TestDeckIO(unittest.TestCase):
    def test_load_base_model(self):
        primary = Path("data/baseline_deck/HEAD_OF_MODEL_1.DATA")
        fallback = Path("Материалы для работы/HEAD_OF_MODEL_1.DATA")
        base_path = primary if primary.is_file() else fallback
        self.assertTrue(base_path.is_file())
        model = DeckParser.load_reservoir_model(base_path)
        self.assertEqual(model["dimens"], (15, 15, 6))
        for prop in ["PORO", "PERMX", "PERMY", "PERMZ", "NTG"]:
            self.assertIn(prop, model["properties"])
            self.assertEqual(model["properties"][prop].shape, (6, 15, 15))


class TestGeostatistics(unittest.TestCase):
    def test_grf_shape_and_standardization(self):
        shape = (6, 15, 15)
        field = GeostatisticalEngine.generate_gaussian_random_field_3d(shape, rx=3.0, ry=3.0, rz=2.0, seed=123)
        self.assertEqual(field.shape, shape)
        self.assertAlmostEqual(float(field.mean()), 0.0, places=5)
        self.assertAlmostEqual(float(field.std()), 1.0, places=3)

    def test_vertical_correlation_preservation(self):
        shape = (6, 15, 15)
        # With vertical correlation rz=3.0
        field_correlated = GeostatisticalEngine.generate_gaussian_random_field_3d(shape, rx=3.0, ry=3.0, rz=3.0, seed=42)
        lags, gamma_z = GeostatisticalEngine.compute_experimental_variogram(field_correlated, "z")
        # With rz=3.0, lag 1 semi-variance should be well below total variance (~1.0)
        self.assertLess(gamma_z[0], 0.60)


class TestPetrophysics(unittest.TestCase):
    def test_cosimulation_correlation(self):
        poro = np.random.uniform(0.10, 0.28, size=(6, 15, 15))
        props = PetrophysicsEngine.cosimulate_properties(poro, corr_factor=0.92, aniso_z=0.25, seed=999)
        kx = props["PERMX"]
        corr = np.corrcoef(poro.flatten(), np.log10(kx.flatten()))[0, 1]
        self.assertGreater(corr, 0.80)
        # Check anisotropy
        ratio_z = np.mean(props["PERMZ"] / props["PERMX"])
        self.assertAlmostEqual(ratio_z, 0.25, delta=0.08)


class TestProxyFlow(unittest.TestCase):
    def test_simulation_runs(self):
        sim = ReservoirProxySimulator(dimens=(15, 15, 6))
        poro = np.full((6, 15, 15), 0.18)
        perm = np.full((6, 15, 15), 100.0)
        res = sim.simulate(poro, perm, perm * 0.8, perm * 0.2, timesteps_days=[30.0, 60.0])
        self.assertIn("field", res)
        self.assertEqual(len(res["field"]["FOPR"]), 2)
        self.assertGreater(res["field"]["FOPR"][0], 0.0)


class TestValidator(unittest.TestCase):
    def test_validation_pass(self):
        poro = np.full((6, 15, 15), 0.18)
        perm = np.full((6, 15, 15), 100.0)
        props = {
            "PORO": poro,
            "PERMX": perm,
            "PERMY": perm * 0.7,
            "PERMZ": perm * 0.2,
            "NTG": np.full((6, 15, 15), 0.85)
        }
        report = ReservoirValidator.validate_property_set(props)
        self.assertEqual(report["status"], "PASS")
        self.assertGreaterEqual(report["score"], 80)


class TestEndToEndPhysicsAndAPI(unittest.TestCase):
    def test_petrophysics_exact_correlation_control(self):
        poro = GeostatisticalEngine.generate_gaussian_random_field_3d((6, 15, 15), rx=4.0, ry=4.0, rz=2.5, seed=7)
        poro = np.clip(0.18 + 0.04 * poro, 0.05, 0.35)
        props = PetrophysicsEngine.cosimulate_properties(poro, corr_factor=0.90, aniso_z=0.20, seed=11)
        r = np.corrcoef(poro.flatten(), np.log10(props["PERMX"].flatten()))[0, 1]
        self.assertGreater(r, 0.85)
        self.assertLess(r, 0.95)

    def test_proxy_flow_distinct_and_physical_water_cut(self):
        sim = ReservoirProxySimulator(dimens=(15, 15, 6))
        p1 = np.full((6, 15, 15), 0.16)
        k1 = np.full((6, 15, 15), 40.0)
        p2 = np.full((6, 15, 15), 0.24)
        k2 = np.full((6, 15, 15), 140.0)
        r1 = sim.simulate(p1, k1, k1 * 0.7, k1 * 0.2)
        r2 = sim.simulate(p2, k2, k2 * 0.7, k2 * 0.2)
        self.assertEqual(r1["field"]["FWCT"][0], 0.0)
        self.assertGreater(r1["field"]["FWCT"][-1], r1["field"]["FWCT"][0])
        self.assertFalse(np.allclose(r1["field"]["FOPR"], r2["field"]["FOPR"]))

    def test_fastapi_endpoints(self):
        from fastapi.testclient import TestClient
        from web.app import app

        client = TestClient(app)
        self.assertEqual(client.get("/favicon.ico").status_code, 204)
        sys_resp = client.get("/api/system")
        self.assertEqual(sys_resp.status_code, 200)
        self.assertTrue(sys_resp.json()["base_model"]["loaded"])
        self.assertEqual(client.get("/api/base-3d?property_name=PORO").status_code, 200)


if __name__ == "__main__":
    unittest.main()
