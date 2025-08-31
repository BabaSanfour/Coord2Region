from io import BytesIO

from PIL import Image

from coord2region.image_utils import add_watermark


def test_add_watermark_modifies_image_bytes():
    img = Image.new("RGB", (100, 100), color="red")
    buf = BytesIO()
    img.save(buf, format="PNG")
    original = buf.getvalue()

    watermarked = add_watermark(original, text="WM")

    assert isinstance(watermarked, bytes)
    assert watermarked != original
