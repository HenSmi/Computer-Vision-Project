import numpy as np
from PIL import Image
from scipy.ndimage import correlate1d

COLOR_BGR2GRAY = 6
COLOR_BGR2RGB = 4
COLOR_Lab2RGB = 100
IMREAD_GRAYSCALE = 0


def imread(path, flags=1):
    image = Image.open(path)
    if flags == IMREAD_GRAYSCALE:
        image = image.convert("L")
        return np.asarray(image, dtype=np.uint8)

    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if flags == 1:
        return array[:, :, ::-1]
    return array


def cvtColor(image, code):
    if code == COLOR_BGR2GRAY:
        rgb = image[:, :, ::-1].astype(np.float32)
        gray = (
            0.299 * rgb[:, :, 0]
            + 0.587 * rgb[:, :, 1]
            + 0.114 * rgb[:, :, 2]
        )
        return gray.astype(np.uint8)

    if code == COLOR_BGR2RGB:
        return image[:, :, ::-1]

    if code == COLOR_Lab2RGB:
        lab = image.astype(np.float32)
        lab = lab.copy()
        lab[..., 0] = np.clip(lab[..., 0] * 100.0 / 255.0, 0.0, 100.0)
        lab[..., 1] = np.clip(lab[..., 1] - 128.0, -128.0, 127.0)
        lab[..., 2] = np.clip(lab[..., 2] - 128.0, -128.0, 127.0)

        L = lab[..., 0]
        a = lab[..., 1]
        b = lab[..., 2]

        def f_lab(t):
            delta = 6.0 / 29.0
            delta_sq = delta ** 2
            delta_cu = delta ** 3
            return np.where(t > delta_cu, np.cbrt(t), t / (3 * delta_sq) + 4.0 / 29.0)

        fy = f_lab((L + 16) / 116.0)
        fx = f_lab((a / 500.0) + fy)
        fz = f_lab(fy - (b / 200.0))

        x = fx
        y = fy
        z = fz

        xyz = np.stack([x, y, z], axis=-1)
        d65_ref = np.array([0.95047, 1.00000, 1.08883], dtype=np.float32)
        xyz = xyz / d65_ref

        r_linear = 3.2404542 * xyz[..., 0] - 1.5371385 * xyz[..., 1] - 0.4985314 * xyz[..., 2]
        g_linear = -0.9692660 * xyz[..., 0] + 1.8760108 * xyz[..., 1] + 0.0415560 * xyz[..., 2]
        b_linear = 0.0556434 * xyz[..., 0] - 0.2040259 * xyz[..., 1] + 1.0572252 * xyz[..., 2]

        srgb = np.stack([
            np.where(r_linear <= 0.0031308, 12.92 * r_linear, 1.055 * np.power(r_linear, 1.0 / 2.4) - 0.055),
            np.where(g_linear <= 0.0031308, 12.92 * g_linear, 1.055 * np.power(g_linear, 1.0 / 2.4) - 0.055),
            np.where(b_linear <= 0.0031308, 12.92 * b_linear, 1.055 * np.power(b_linear, 1.0 / 2.4) - 0.055),
        ], axis=-1)

        rgb = np.clip(srgb, 0.0, 1.0)
        return (rgb * 255.0).astype(np.uint8)

    raise NotImplementedError(f"Unsupported colour conversion code: {code}")


def bitwise_and(a, b):
    return np.bitwise_and(a, b)

# def GaussianBlur(
#     image: np.ndarray,
#     ksize: tuple[int, int],
#     sigma: float = 0
# ) -> np.ndarray:
#     from scipy.ndimage import convolve
    
#     # compute sigma if 0
#     if sigma == 0:
#         sigma = 0.3 * ((ksize[0] - 1) * 0.5 - 1) + 0.8
    
#     # Create 1D Gaussian kernel
#     radius = ksize[0] // 2
#     x = np.arange(-radius, radius + 1)      # needed to seperate so that the -radius was not (for example) -3 for 5,5 kernel
#     kernel_1d = np.exp(-(x**2) / (2 * sigma**2))
#     kernel_1d /= kernel_1d.sum()
#     # Convolve image with 1D kernel on each channel
#     result = np.zeros_like(image, dtype=np.float32) 
    
#     img = image.astype(np.float32)
#     img = correlate1d(img, kernel_1d, axis=0, mode='reflect')
#     img = correlate1d(img, kernel_1d, axis=1, mode='reflect')
    
#     # Clip and convert back to original dtype
#     result = np.clip(result, 0, 255)
#     return result.astype(image.dtype)

def GaussianBlur(image, ksize, sigma=0):
    # compute sigma if 0
    if sigma == 0:
        sigma = 0.3 * ((ksize[0] - 1) * 0.5 - 1) + 0.8

    # Create 1D Gaussian kernel
    radius = ksize[0] // 2
    x = np.arange(-radius, radius + 1) # needed to seperate so that the -radius was not (for example) -3 for 5,5 kernel
    kernel_1d = np.exp(-(x**2) / (2 * sigma**2))
    kernel_1d /= kernel_1d.sum()

    img = image.astype(np.float32)
    img = correlate1d(img, kernel_1d, axis=0, mode='reflect')   #much faster than stepping through the image
    img = correlate1d(img, kernel_1d, axis=1, mode='reflect')

    return np.clip(img, 0, 255).astype(image.dtype)

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

def BGR2LABone(bgr: np.ndarray) -> np.ndarray:
    rgb = np.array([0,0,0])
    rgb[0] = bgr[2]
    rgb[1] = bgr[1]
    rgb[2] = bgr[0]
    # Normalize to [0, 1] as that is what is needed for converting to XYZ
    rgb_norm = rgb / 255.0
    
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
    xyz = [0,0,0]
    for i in range(3):
        for j in range(3):
            xyz[i] += rgb_linear[j] * matrix_rgb2xyz[i][j]
    
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
    
    f_x = f_lab(xyz_norm[0])
    f_y = f_lab(xyz_norm[1])
    f_z = f_lab(xyz_norm[2])
    
    # Compute LAB
    L = 116 * f_y - 16
    a = 500 * (f_x - f_y)
    b = 200 * (f_y - f_z)
    # print(L,a,b)
    # convert to 255 mapping
    # L_uint8 = (L * 255.0 / 100.0).astype(np.uint8)
    # a_uint8 = (a + 128).clip(0, 255).astype(np.uint8)
    # b_uint8 = (b + 128).clip(0, 255).astype(np.uint8)
    
    lab = np.array([L,a,b])
    return lab

# print(BGR2LABone(np.array([ 35,  94, 124])))
# print(BGR2LABone(np.array([ 200,  200, 20])))