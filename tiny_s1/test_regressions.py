"""Regression tests for output identity, input limits and numerical stability."""
import copy
import json
import unittest
import numpy as np
import autograd as ag
import data as D
from model import TinyS1
from api import example_payload, predict, request_to_internal


class Regressions(unittest.TestCase):
    def setUp(self):
        ag.set_dtype(np.float64)
        self.model = TinyS1(len(D.VOCAB), seed=2)

    def test_serialized_probabilities_keep_every_hidden_option(self):
        req = D.sample_request(D.make_rng(19), name_mode="hidden")
        answer = self.model.answer(D.collate([req]))[0]
        self.assertEqual(set(answer["probabilities"]), set(req["questions"][0]["offered"]))
        self.assertAlmostEqual(sum(answer["probabilities"].values()), 1.0, places=14)
        json.dumps(answer, allow_nan=False)

    def test_duplicate_names_without_output_ids_are_rejected(self):
        req = D.sample_request(D.make_rng(19), name_mode="hidden")
        req["questions"][0].pop("option_ids")
        with self.assertRaises(ValueError):
            self.model.answer(D.collate([req]))

    def test_api_subset_and_single_question_types(self):
        payload = example_payload()
        response = predict(self.model, payload)
        self.assertEqual(list(response["answers"]), list(payload["questions"]))
        for name, q in payload["questions"].items():
            single = dict(payload, questions={name: q})
            result = predict(self.model, single)["answers"][name]
            if q["type"] == "noul":
                self.assertAlmostEqual(result["noul"], response["answers"][name]["noul"], places=13)
            else:
                self.assertAlmostEqual(sum(result["probabilities"].values()), 1, places=14)
                for k, p in result["probabilities"].items():
                    self.assertAlmostEqual(p, response["answers"][name]["probabilities"][k], places=13)
                if q["type"] == "score":
                    self.assertIn("legend", result)
                    self.assertAlmostEqual(result["score"], sum(int(k) * p for k, p in result["probabilities"].items()))
            loss, _ = self.model.loss(D.collate([request_to_internal(single)]))
            self.assertTrue(np.isfinite(loss.data))

    def test_limits_fail_clearly(self):
        payload = example_payload()
        payload["questions"]["urgency"]["criteria"] = ["low"]
        with self.assertRaises(ValueError):
            predict(self.model, payload)
        for key, count in [("state", D.MAX_STATE), ("instr", D.MAX_Q)]:
            req = D.sample_request(D.make_rng(1))
            if key == "state":
                req[key] = ["help"] * count
            else:
                req["questions"][0][key] = ["help"] * count
            with self.assertRaises(ValueError):
                D.collate([req])

    def test_single_type_training_backward(self):
        request = D.sample_request(D.make_rng(23))
        for question in request['questions']:
            single = {'state': request['state'], 'questions': [question]}
            for p in self.model.parameters():
                p.grad = None
            loss, _ = self.model.loss(D.collate([single]))
            loss.backward()
            for p in self.model.parameters():
                if p.grad is not None:
                    self.assertTrue(np.isfinite(p.grad).all())

    def test_extreme_log_loss_and_derivative_agree(self):
        s = ag.Tensor(np.array([1000., -1000.]), requires_grad=True)
        loss = ag.grouped_softmax_xent(s, np.array([0, 0]), np.array([0., 1.]), 1)
        self.assertEqual(float(loss.data), 2000.)
        loss.backward()
        np.testing.assert_array_equal(s.grad, [1., -1.])
        with np.errstate(over="raise"):
            np.testing.assert_array_equal(ag.sigmoid_np(np.array([-1000., 1000.])), [0., 1.])

    def test_positive_finite_temperature(self):
        batch = D.collate([D.sample_request(D.make_rng(1))])
        for t in (0, -1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                self.model.answer(batch, t_opt=t)

    def test_sibling_options_cannot_change_shared_log_odds(self):
        req = D.sample_request(D.make_rng(4), n_opts=(4, 4), name_mode="topic")
        q = req["questions"][0]
        small = copy.deepcopy(req)
        for key in ("options", "option_ids", "target"):
            small["questions"][0][key] = small["questions"][0][key][:2]
        scores_large, _ = self.model.forward(D.collate([req]))
        scores_small, _ = self.model.forward(D.collate([small]))
        self.assertAlmostEqual(float(scores_large.data[0] - scores_large.data[1]),
                               float(scores_small.data[0] - scores_small.data[1]), places=12)

    def test_lexical_probability_scale_and_ranking(self):
        from baselines import design, distribution
        counts = np.array([[1., 0., 0., 0.], [1., 0., 0., 0.]])
        n = np.array([2, 4])
        beta = np.array([np.log(.7 / .3), 1.])
        p, _ = distribution(beta, design(counts, n, True), n)
        np.testing.assert_allclose(p, [[.7, .3, 0., 0.], [.7, .1, .1, .1]], atol=1e-14)
        fixed, _ = distribution(np.array([1.8]), design(counts, n, False), n)
        self.assertAlmostEqual(fixed[0, 0] / fixed[0, 1], fixed[1, 0] / fixed[1, 1], places=13)
        np.testing.assert_array_equal(p.argmax(axis=1), fixed.argmax(axis=1))

    def test_lexical_fit_is_numerically_stationary(self):
        from baselines import features, fit, design, distribution
        rng = D.make_rng(177)
        counts, n, y, _ = features([D.sample_request(rng) for _ in range(512)])
        beta, _ = fit(counts, n, y, True)
        x = design(counts, n, True)
        def objective(parameters):
            return -distribution(parameters, x, n)[1][np.arange(len(y)), y].mean()
        for i in range(2):
            delta = np.zeros(2); delta[i] = 1e-5
            numerical = (objective(beta + delta) - objective(beta - delta)) / (2e-5)
            self.assertLess(abs(numerical), 1e-7)


if __name__ == "__main__":
    unittest.main(verbosity=2)
