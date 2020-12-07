import numpy as np

class FTSmoother(object):
    """
    A simple interface class for smoothing FT sensor values

    """
    DISABLE = 0
    LOCAL = 1
    GLOBAL = 2

    def __init__(self, ft_function_handle, measurement_frame, buffer_length=50):
        self._frame_type = measurement_frame
        self._ft_handle = ft_function_handle

        self._buffer_len = buffer_length

        self._running_avg_buffer = np.zeros([6, self._buffer_len])

        self._count = 0

    def update(self, values=None):
        if self._frame_type == self.DISABLE:
            print ("Not updating FT smoother. Frame not specified.")
            return

        if values is None:
            self._running_avg_buffer[:, self._count % self._buffer_len] = self._ft_handle(
                local=(self._frame_type == self.LOCAL))
        else:
            self._running_avg_buffer[:, self._count % self._buffer_len] = values

        self._count += 1

    def get_values(self):

        if self._frame_type == self.DISABLE:
            raise ValueError("FT Smoother not properly defined")

        if self._count >= self._buffer_len:
            return np.mean(self._running_avg_buffer, 1)

        return np.mean(self._running_avg_buffer[:, :self._count], 1)
