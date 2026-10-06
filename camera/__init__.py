"""Camera input: threaded webcam capture and a simulated camera."""

import os
import sys

if sys.platform.startswith("win"):
    # Media Foundation takes ~8 s to open a camera unless hardware video transforms are
    # disabled.  OpenCV reads this variable when it first initialises, so it has to be set
    # *before* the first ``import cv2`` anywhere in the process - hence it lives here and at
    # the very top of app.py.
    os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")
