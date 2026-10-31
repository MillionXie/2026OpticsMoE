import pytest
import torch
from LightGenV2.tasks.t17_router_classification.original_f_teacher import training_view_codes,transform_teacher_images,teacher_view_indices
from LightGenV2.tasks.t17_router_classification.train_figure_revision import augment_packed
from LightGenV2.tasks.t17_router_classification.model_four import encode_rgb


@pytest.mark.parametrize('code',range(8))
def test_teacher_raw_image_geometry_matches_student_channel_tile_geometry(code):
    torch.set_num_threads(4);image=torch.randint(0,256,(2,28,28,3),dtype=torch.uint8)
    actual=encode_rgb(transform_teacher_images(image,code))
    expected=augment_packed(encode_rgb(image),torch.full((2,),code,dtype=torch.long))
    assert torch.allclose(actual,expected,rtol=1e-4,atol=1e-7)
    assert teacher_view_indices(torch.tensor([code]),'D4').item()==code


def test_existing_hflip_view_mapping_stays_exact_and_D4_is_explicit():
    images=torch.randint(0,256,(2,56,56,3),dtype=torch.uint8)
    assert training_view_codes('horizontal_flip')==(0,4)
    assert torch.equal(transform_teacher_images(images,0),images)
    assert torch.equal(transform_teacher_images(images,4),images.flip(2))
    assert teacher_view_indices(torch.tensor([0,4]),'horizontal_flip').tolist()==[0,1]
    assert training_view_codes('D4')==tuple(range(8))


def test_uncached_or_invalid_augmentations_cannot_silently_use_wrong_teacher_target():
    with pytest.raises(ValueError):teacher_view_indices(torch.tensor([1]),'horizontal_flip')
    with pytest.raises(ValueError):training_view_codes('unknown')
    with pytest.raises(ValueError):transform_teacher_images(torch.ones(1,4,4,3),8)
