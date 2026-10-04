import os
import sys

from absl.testing import absltest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import moe_tune  # noqa: E402


class FakeRayTest(absltest.TestCase):
    def test_decorated_actor_runs_calls_synchronously(self):
        ray = moe_tune.fake_ray()

        @ray.remote(num_gpus=1)
        class Worker:
            def __init__(self, seed):
                self.seed = seed
                self.gpu = ray.get_gpu_ids()[0]

            def tune(self, m):
                return (self.seed, self.gpu, m)

        ray.init()
        workers = [Worker.remote(7) for _ in range(int(ray.available_resources()["GPU"]))]
        self.assertLen(workers, 1)
        out = ray.get([workers[0].tune.remote(m) for m in (1, 2)])
        self.assertEqual(out, [(7, 0, 1), (7, 0, 2)])

    def test_bare_decorator(self):
        ray = moe_tune.fake_ray()

        @ray.remote
        class W:
            def f(self):
                return 3

        self.assertEqual(ray.get(W.remote().f.remote()), 3)

    def test_tqdm_import_path(self):
        moe_tune.install_fake_ray()
        from ray.experimental.tqdm_ray import tqdm

        self.assertEqual(list(tqdm([1, 2], disable=True)), [1, 2])


if __name__ == "__main__":
    absltest.main()
