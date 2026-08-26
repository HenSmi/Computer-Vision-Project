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
    radius = ksize[0] // 2
    x = np.arange(-radius, radius + 1)      # needed to seperate so that the -radius was not (for example) -3 for 5,5 kernel
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

#to create mask that maps what pixels are actually "bean colours"
def inRange(
    image: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray
) -> np.ndarray:
    # Check all channels at same time
    in_range = np.all((image >= lower) & (image <= upper), axis=2)
    
    # Convert boolean to uint8 (0 or 255)
    return (in_range * 255).astype(np.uint8)

def BGR2LAB(image: np.ndarray) -> np.ndarray:
    # BGR → RGB (flip channels) this will be removed if i no longer use opencv.imread
    rgb = image[:, :, ::-1]
    
    # Normalize to [0, 1] as that is what is needed for converting to XYZ
    rgb_norm = rgb.astype(np.float64) / 255.0
    
    # Apply gamma correction (sRGB → linear RGB) as specified by standard
    rgb_linear = np.where(
        rgb_norm <= 0.04045,
        rgb_norm / 12.92,
        np.power((rgb_norm + 0.055) / 1.055, 2.4)
    )
    
    # Matrix for converting linear RGB to XYZ
    matrix_rgb2xyz = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ])
    
    # Apply transformation RGB_linear → XYZ using D65 matrix
    h, w = rgb_linear.shape[:2]
    xyz = np.zeros((h, w, 3), dtype=np.float64)
    for c in range(3):
        for i in range(3):
            xyz[:, :, c] += rgb_linear[:, :, i] * matrix_rgb2xyz[c, i]
    
    # Normalize by D65 reference white as per CIE 1931 for 2 degree observer
    d65_ref = np.array([0.95047, 1.00000, 1.08883])
    xyz_norm = xyz / d65_ref
    
    # XYZ_norm → LAB using f function
    delta = 6.0 / 29.0
    delta_sq = delta ** 2
    delta_cu = delta ** 3

    # search through all pixels and map according to f()
    def f_lab(t):
        return np.where(
            t > delta_cu,
            np.power(t, 1.0/3.0),
            t / (3 * delta_sq) + 4.0/29.0
        )
    
    f_x = f_lab(xyz_norm[:, :, 0])
    f_y = f_lab(xyz_norm[:, :, 1])
    f_z = f_lab(xyz_norm[:, :, 2])
    
    # Compute LAB
    L = 116 * f_y - 16
    a = 500 * (f_x - f_y)
    b = 200 * (f_y - f_z)
    
    # convert to 255 mapping
    L_uint8 = (L * 255.0 / 100.0).astype(np.uint8)
    a_uint8 = (a + 128).clip(0, 255).astype(np.uint8)
    b_uint8 = (b + 128).clip(0, 255).astype(np.uint8)
    
    lab = np.stack([L_uint8, a_uint8, b_uint8], axis=2)
    return lab