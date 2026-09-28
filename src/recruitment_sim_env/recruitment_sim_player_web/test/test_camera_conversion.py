import pytest
from sensor_msgs.msg import Image

from recruitment_sim_player_web.server import ros_image_to_video_frame


@pytest.mark.parametrize("step", [6, 8])
def test_camera_frame_with_packed_or_padded_rows(step):
    message = Image()
    message.width = 2
    message.height = 2
    message.encoding = "rgb8"
    message.step = step
    padding = [9] * (step - 6)
    message.data = [255, 0, 0, 0, 255, 0] + padding + [
        0, 0, 255, 255, 255, 255,
    ] + padding

    frame = ros_image_to_video_frame(message)

    assert (frame.width, frame.height) == (2, 2)
    assert frame.to_ndarray(format="rgb24").tolist() == [
        [[255, 0, 0], [0, 255, 0]],
        [[0, 0, 255], [255, 255, 255]],
    ]


def test_rejects_short_camera_buffer():
    message = Image()
    message.width = 1
    message.height = 1
    message.encoding = "rgb8"
    message.step = 3
    message.data = [1, 2]

    with pytest.raises(ValueError, match="shorter"):
        ros_image_to_video_frame(message)


def test_aligned_camera_frame_pixels():
    message = Image()
    message.width = 16
    message.height = 1
    message.encoding = "rgb8"
    message.step = 48
    message.data = list(range(48))

    frame = ros_image_to_video_frame(message)

    assert frame.planes[0].line_size == 48
    assert frame.to_ndarray(format="rgb24").tobytes() == bytes(range(48))
