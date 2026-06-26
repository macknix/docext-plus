"""Tests for document discovery and TIFF->JPEG conversion."""
from __future__ import annotations

import pytest
from PIL import Image

from docext_batch.images import list_documents
from docext_batch.images import set_max_image_pixels
from docext_batch.images import tiff_to_jpeg


def test_list_documents_filters_and_sorts(tmp_path):
    (tmp_path / "b.tiff").touch()
    (tmp_path / "a.tif").touch()
    (tmp_path / "note.txt").touch()
    (tmp_path / "sub").mkdir()

    found = list_documents(tmp_path, (".tif", ".tiff"))
    assert [p.name for p in found] == ["a.tif", "b.tiff"]  # sorted, extension-filtered


def test_list_documents_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        list_documents(tmp_path / "nope", (".tiff",))


@pytest.mark.parametrize("mode", ["L", "1", "RGB"])  # grayscale, bitonal, colour
def test_tiff_to_jpeg_converts_and_downscales(tmp_path, mode):
    src = tmp_path / "scan.tiff"
    Image.new(mode, (4000, 3000)).save(src)

    out = tiff_to_jpeg(src, tmp_path / "work", max_size=2048)

    assert out.suffix == ".jpg"
    with Image.open(out) as im:
        assert im.format == "JPEG"
        assert im.mode == "RGB"          # always flattened to RGB
        assert max(im.size) == 2048      # longest side downscaled, aspect kept
        assert im.size == (2048, 1536)


def test_tiff_to_jpeg_rotate_90_swaps_dimensions(tmp_path):
    src = tmp_path / "scan.tiff"
    Image.new("RGB", (100, 60)).save(src)  # wide
    out = tiff_to_jpeg(src, tmp_path / "work", max_size=2048, rotate=90)
    with Image.open(out) as im:
        assert im.size == (60, 100)  # 90° rotation swaps width/height (expand=True)


def test_tiff_to_jpeg_no_rotate_keeps_dimensions(tmp_path):
    src = tmp_path / "scan.tiff"
    Image.new("RGB", (100, 60)).save(src)
    out = tiff_to_jpeg(src, tmp_path / "work", max_size=2048, rotate=0)
    with Image.open(out) as im:
        assert im.size == (100, 60)


def test_tiff_to_jpeg_no_upscale(tmp_path):
    src = tmp_path / "small.tiff"
    Image.new("RGB", (100, 80)).save(src)
    out = tiff_to_jpeg(src, tmp_path / "work", max_size=2048)
    with Image.open(out) as im:
        assert im.size == (100, 80)  # smaller than max_size -> untouched


def test_set_max_image_pixels():
    set_max_image_pixels(123)
    assert Image.MAX_IMAGE_PIXELS == 123
    set_max_image_pixels(None)
    assert Image.MAX_IMAGE_PIXELS is None
    set_max_image_pixels(300_000_000)  # restore a sane default for other tests
