import unittest
from collections import Counter
from PIL import Image
from LightGenV2.tasks.t12_text_to_image.scene_paired_corpus import background_requests, composite


class CorpusContractTest(unittest.TestCase):
    def test_assets_have_independent_split_seeds(self):
        rows = background_requests()
        self.assertEqual(Counter(r['split'] for r in rows), {'train': 32, 'dev': 16})
        self.assertEqual(len({r['seed'] for r in rows}), 48)
        self.assertFalse(any(r['split'] == 'test' for r in rows))

    def test_transparent_rgb_does_not_create_fringe(self):
        product = Image.new('RGBA', (20, 20), (255, 255, 255, 0))
        for y in range(5, 15):
            for x in range(5, 15): product.putpixel((x, y), (30, 80, 120, 255))
        result, alpha, box = composite(product, Image.new('RGB', (1024, 1024), (90, 90, 90)), 'pillow')
        center = ((box[0]+box[2])//2, (box[1]+box[3])//2)
        self.assertEqual(result.getpixel(center), (30, 80, 120))
        self.assertEqual(result.getpixel((0, 0)), (90, 90, 90))
        self.assertEqual(alpha.getbbox(), tuple(box))


if __name__ == '__main__': unittest.main()
