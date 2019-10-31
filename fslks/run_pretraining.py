from absl import logging
from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_multi_string("tasks", None, "One or more tasks to be used for pretraining")


def main(argv):
    del argv  # Unused.
    raise NotImplementedError()


if __name__ == '__main__':
    app.run(main)
