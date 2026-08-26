import numpy as np

def GaussianBlur(
    image: np.ndarray,
    ksize: tuple[int, int],
    sigma: float = 0
) -> np.ndarray:
    from scipy.ndimage import convolve
    
    # compute sigma if 0
    if sigma == 0:
        sigma = 0.3 * ((ksize[0] - 1) * 0.5 - 1) + 0.8
    
    # Create 1D Gaussian kernel
    x = np.linspace(-ksize[0]//2, ksize[0]//2, ksize[0])
    kernel_1d = np.exp(-(x**2) / (2 * sigma**2))
    kernel_1d /= kernel_1d.sum()
    
    # Convolve image with 1D kernel on each channel
    result = np.zeros_like(image, dtype=np.float32)
    
    for c in range(image.shape[2]):
        # Convolve horizontal kernel
        temp = convolve(
            image[:, :, c].astype(np.float32),
            kernel_1d[:, np.newaxis],
            mode='reflect'
        )
        # Convolve vertical kernel
        result[:, :, c] = convolve(
            temp,
            kernel_1d[np.newaxis, :],
            mode='reflect'
        )
    
    # Clip and convert back to original dtype
    result = np.clip(result, 0, 255)
    return result.astype(image.dtype)