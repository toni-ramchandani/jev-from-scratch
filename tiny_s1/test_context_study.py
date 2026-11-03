"""Gradient, information-flow and generator checks for context experiments."""
import copy
import hashlib
import json
import unittest
import numpy as np
import autograd as ag
import data as D
from model import TinyS1
from context_study import ContextModel, choice_request, make_batch


class ContextChecks(unittest.TestCase):
    def setUp(self):
        ag.set_dtype(np.float64)

    def tearDown(self):
        ag.set_dtype(np.float32)

    def test_default_generator_unchanged(self):
        rng = D.make_rng(98371)
        requests = [D.sample_request(rng) for _ in range(32)]
        text = json.dumps(requests, sort_keys=True, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
        self.assertEqual(hashlib.sha256(text.encode()).hexdigest(),
                         'a1fafbce4c6d4f9a2b92b2388c1f41dfd31172488233aa4b0c9890f3ea4a14c5')

    def test_control_world_has_constant_likelihood_ratio(self):
        for n in range(2, 7):
            rng = D.make_rng(n)
            req = choice_request(rng, 'fixed_odds', n_opts=(n, n))
            q = req['questions'][0]
            counts = np.array([sum(w in description for w in req['state']) for _, description in q['options']])
            expected = 7.**counts; expected /= expected.sum()
            np.testing.assert_allclose(q['bayes'], expected, atol=1e-14)

    def test_isolated_initialization_matches_original(self):
        base = TinyS1(len(D.VOCAB), seed=17)
        model = ContextModel(len(D.VOCAB), seed=17)
        req = choice_request(D.make_rng(8), 'fixed_q')
        for left, right in zip(base.parameters(), model.parameters()):
            np.testing.assert_array_equal(left.data, right.data)
        for mode in ('isolated', 'count', 'random_count'):
            batch = make_batch([req], mode, [4])
            np.testing.assert_array_equal(base.forward(D.collate([req]))[0].data, model.forward(batch)[0].data)
        self.assertEqual(sum(p.data.size for p in model.parameters()), 116993)

    def test_set_context_preserves_local_option_identity(self):
        req = D.sample_request(D.make_rng(8))
        b = make_batch([req], 'set')
        kinds, qid, oid = b['kind'][0], b['qid'][0], b['oid'][0]
        first = np.flatnonzero((kinds == D.OPTION) & (qid == 0) & (oid == 0))[0]
        sibling = np.flatnonzero((kinds == D.OPTION) & (qid == 0) & (oid == 1))[0]
        question = np.flatnonzero((kinds == D.QUESTION) & (qid == 0))[0]
        other = np.flatnonzero(qid == 1)[0]
        self.assertEqual(b['mask_bias'][0,0,question,sibling], 0)
        self.assertLess(b['mask_bias'][0,0,first,sibling], -1e8)
        self.assertLess(b['mask_bias'][0,0,first,other], -1e8)
        model = ContextModel(len(D.VOCAB), d=8, n_heads=2, n_layers=2, d_ff=16, seed=3)
        scores, _ = model.forward(b)
        count = len(req['questions'][0]['options'])
        self.assertGreater(float(np.ptp(scores.data[:count])), 1e-8)

    def test_context_embedding_gradient(self):
        model = ContextModel(len(D.VOCAB), d=8, n_heads=2, n_layers=1, d_ff=16, seed=11)
        requests = [choice_request(D.make_rng(i), 'fixed_q') for i in (7, 8)]
        batch = make_batch(requests, 'count')
        loss, _ = model.loss(batch)
        loss.backward()
        numeric = np.zeros_like(model.context.data)
        for index in np.ndindex(numeric.shape):
            old = model.context.data[index]
            model.context.data[index] = old + 1e-6
            plus = float(model.loss(batch)[0].data)
            model.context.data[index] = old - 1e-6
            minus = float(model.loss(batch)[0].data)
            model.context.data[index] = old
            numeric[index] = (plus-minus)/2e-6
        np.testing.assert_allclose(model.context.grad, numeric, rtol=3e-5, atol=1e-8)

    def test_question_isolation_and_option_symmetry(self):
        req = D.sample_request(D.make_rng(7))
        for mode in ('isolated', 'count', 'random_count', 'set'):
            model = ContextModel(len(D.VOCAB), d=8, n_heads=2, n_layers=2, d_ff=16, seed=13)
            model.context.data[:] = np.random.default_rng(4).normal(0, .02, model.context.shape)
            context = [2+i for i in range(len(req['questions']))]
            batch = make_batch([req], mode, context)
            together = model.answer(batch)
            for i, q in enumerate(req['questions']):
                single = model.answer(make_batch([{'state':req['state'], 'questions':[q]}], mode, [context[i]]))[0]
                if q['type'] == 'noul':
                    self.assertAlmostEqual(single['noul'], together[i]['noul'], places=12)
                else:
                    np.testing.assert_allclose(list(single['probabilities'].values()),
                                               list(together[i]['probabilities'].values()), atol=1e-12)
            shuffled = copy.deepcopy(req)
            q = shuffled['questions'][0]
            perm = list(reversed(range(len(q['options']))))
            for key in ('options','option_ids','target'):
                q[key] = [q[key][i] for i in perm]
            reordered = model.answer(make_batch([shuffled], mode, context))[0]['probabilities']
            for key, value in together[0]['probabilities'].items():
                self.assertAlmostEqual(value, reordered[key], places=12)


if __name__ == '__main__':
    unittest.main(verbosity=2)
