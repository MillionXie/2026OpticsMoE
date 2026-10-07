from LightGenV2.tasks.t12_text_to_image.feature_cache import _deduplicate_captions


def test_caption_deduplication_preserves_first_seen_order_and_inverse() -> None:
    captions = ["red shirt", "blue coat", "red shirt", "green dress", "blue coat"]
    unique, inverse = _deduplicate_captions(captions)
    assert unique == ["red shirt", "blue coat", "green dress"]
    assert [unique[index] for index in inverse] == captions
