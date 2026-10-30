from LightGenV2.tasks.t12_text_to_image.scene_lab_mosaic_dll_probe import version_tuple


def test_product_and_build_version_are_distinct():
    v=dict(ProductVersionMS=2<<16,ProductVersionLS=7<<16,FileVersionMS=2<<16,FileVersionLS=8<<16)
    assert version_tuple(v)==(2,0,7,0)
    assert version_tuple(v,'FileVersion')==(2,0,8,0)
