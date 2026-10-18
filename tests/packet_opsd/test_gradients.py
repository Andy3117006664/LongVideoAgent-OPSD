import importlib.util
import math
import unittest

from methods.packet_opsd.gradients import packet_gradient_scale

TORCH = importlib.util.find_spec("torch") is not None
if TORCH:
    import torch
    from methods.packet_opsd.gradients import bounded_packet_gradients


class GradientScaleTest(unittest.TestCase):
    def test_large_auxiliary_is_limited_without_length_normalizing_scores(self):
        scale = packet_gradient_scale(.4647992141, 7.2047231972, .5)
        self.assertAlmostEqual(scale * 7.2047231972, .5 * .4647992141)
        self.assertEqual(packet_gradient_scale(1, .1, .5), 1)
        self.assertEqual(packet_gradient_scale(0, 1, .5), 0)
        for values in ((math.nan, 1, .5), (1, math.inf, .5), (1, 1, -1)):
            with self.assertRaises(ValueError):
                packet_gradient_scale(*values)


@unittest.skipUnless(TORCH, "torch is not installed")
class GradientCombinationTest(unittest.TestCase):
    def test_actual_accumulated_packet_increment_obeys_relative_bound(self):
        parameters = [torch.nn.Parameter(torch.zeros(2)), torch.nn.Parameter(torch.zeros(1))]
        reference = [torch.tensor([.3, .4]), None]
        parameters[0].grad = torch.tensor([30., -40.])
        parameters[1].grad = torch.tensor([12.])
        metrics = bounded_packet_gradients(parameters, reference, max_ratio=.5)
        delta = torch.cat([parameters[0].grad - reference[0], parameters[1].grad])
        self.assertAlmostEqual(float(delta.norm()), .25, places=6)
        self.assertLessEqual(metrics['packet_gradient_ratio'], .500001)
        self.assertTrue(metrics['packet_gradient_clipped'])
        self.assertGreater(metrics['raw_weighted_packet_gradient_norm'], 50)
        # The unnormalized packet direction is retained under a single scale.
        torch.testing.assert_close(delta / metrics['packet_gradient_scale'], torch.tensor([30., -40., 12.]))

    def test_weak_packet_and_zero_reference_do_not_create_a_packet_only_step(self):
        parameter = torch.nn.Parameter(torch.zeros(2))
        parameter.grad = torch.tensor([.01, .02])
        reference = [torch.tensor([1., 0.])]
        metrics = bounded_packet_gradients([parameter], reference, max_ratio=.5)
        self.assertEqual(metrics['packet_gradient_scale'], 1)
        torch.testing.assert_close(parameter.grad, torch.tensor([1.01, .02]))
        parameter.grad = torch.tensor([1., 2.])
        metrics = bounded_packet_gradients([parameter], [None], max_ratio=.5)
        self.assertEqual(metrics['packet_gradient_scale'], 0)
        self.assertEqual(metrics['packet_gradient_norm'], 0)
        self.assertEqual(float(parameter.grad.norm()), 0)

    def test_cancellation_and_unrelated_parameters_are_preserved(self):
        parameter = torch.nn.Parameter(torch.zeros(2))
        other = torch.nn.Parameter(torch.zeros(1))
        other.grad = torch.tensor([42.])
        parameter.grad = torch.tensor([-10., 0.])
        metrics = bounded_packet_gradients([parameter], [torch.tensor([1., 0.])], max_ratio=.5)
        torch.testing.assert_close(parameter.grad, torch.tensor([.5, 0.]))
        self.assertEqual(metrics['packet_kl_gradient_cosine'], -1)
        self.assertEqual(float(other.grad), 42)


if __name__ == '__main__':
    unittest.main()
