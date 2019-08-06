from fastai.basic_train import Learner
from fastai.vision import ImageDataBunch

from fast_rl.util.file_handlers import get_absolute_path


def test_ImageDataBunch_init():
    """
    For understanding various databunches.

    For example, ImageDataBunch in the from folder:

    Src is originally an ImageList, but the following code:

    `src = src.label_from_folder(classes=classes)`

    CHANGES THE CLASS TO A LABELLISTS?!?!?

    In other words, the ImageList is capable of turning into a dataset.

    :return:
    """
    data = ImageDataBunch.from_folder(get_absolute_path('data'), valid_pct=0.5)

    for e in data.train_ds:
        print(e)