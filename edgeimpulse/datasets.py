def load_timeseries():
    """
    Loads the timeseries dataset
    """
    import numpy as np

    # create 5 samples, with 3 axis (sensors)
    samples = np.array(
        [
            [  # sample 1
                [8.81, 0.03, 1.21],
                [9.83, 1.04, 1.27],
                [9.12, 0.03, 1.23],
                [9.14, 2.01, 1.25],
            ],
            [  # sample 2
                [8.81, 0.03, 1.21],
                [9.12, 0.03, 1.23],
                [9.14, 2.01, 1.25],
                [9.14, 2.01, 1.25],
            ],
            [  # sample 3
                [8.81, 0.03, 1.21],
                [8.81, 0.03, 1.21],
                [9.83, 1.04, 1.27],
                [9.14, 2.01, 1.25],
            ],
            [  # sample 4
                [9.81, 0.03, 1.21],
                [8.81, 0.03, 1.21],
                [9.83, 1.04, 1.27],
                [9.14, 2.01, 1.25],
            ],
            [  # sample 5
                [10.81, 0.03, 1.21],
                [8.81, 0.03, 1.21],
                [9.83, 1.04, 1.27],
                [9.14, 2.01, 1.25],
            ],
        ]
    )

    sensors = [
        {"name": "accelX", "units": "ms/s"},
        {"name": "accelY", "units": "ms/s"},
        {"name": "accelZ", "units": "ms/s"},
    ]

    labels = ["up", "down", "down", "up", "down"]
    return (samples, labels, sensors)


def load_gestures():
    """
    Loads the gestures dataset
    """
    pass


def load_keywords():
    """
    Loads the keywords dataset
    """

    pass


def load_faucets():
    """
    Loads the faucet dataset
    """

    pass
