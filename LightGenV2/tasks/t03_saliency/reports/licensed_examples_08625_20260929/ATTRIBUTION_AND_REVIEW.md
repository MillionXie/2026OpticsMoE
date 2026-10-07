# SALICON / COCO CC BY 2.0 图片核验

核验时间（UTC）：2026-09-28T09:09:46+00:00

## 结论

原始 6 张中，6 张通过 COCO 官方 `license=4` 与许可证映射核验；其中 3 张同时通过 SALICON 成员身份和 Flickr 当前许可核验。已补充核验 3 张，最终推荐 6 张。

本清单的“完整通过”定义为：COCO 元数据为 CC BY 2.0、文件出现在 SALICON 官方图片包目录、Flickr 返回相同 photo ID 且当前许可为 CC BY 2.0、COCO JPG 下载成功。它是证据核验结果，不是对所有潜在权利的保证。

## 证据与方法

- COCO 官方下载页：[2014 annotations](https://cocodataset.org/#download)，原始压缩包：[annotations_trainval2014.zip](http://images.cocodataset.org/annotations/annotations_trainval2014.zip)。
- 主要依据是压缩包内 `instances_train2014.json` 和 `instances_val2014.json` 的 `images`、`licenses` 字段；已额外核对它们与同包 captions 文件中的图片记录和许可证映射完全一致。
- 两份文件的许可证映射均实际读出 `{id: 4, name: "Attribution License", url: "http://creativecommons.org/licenses/by/2.0/"}`。原始 HTTP URL 原样保留，HTTPS 的同一许可页面用于阅读；不把 HTTP/HTTPS 差异当作许可冲突。
- SALICON 成员身份来源：[官方 challenge 页面](https://www.salicon.net/challenge)链接的 images.zip。以 HTTP Range 读取其 ZIP 中央目录，得到 train 10,000、val 5,000、test 5,000 张；不是依靠第三方文件名猜测。
- Flickr 验证严格匹配 photo ID，并提取对应 ImageObject JSON-LD 中的 author/creator、license 和原始页面地址，同时检查页面可见 CC 链接。跳转到 Explore 或登录页，即使 HTTP=200，也标记 unavailable。
- `evidence/` 中保留完整网页 HTML、请求结果、校验哈希、SALICON 完整文件列表和官方标注压缩包。JPG 是 COCO 提供的原始文件字节，未重新编码；它们不一定等于摄影者相机输出的最高分辨率原片。

## A. 完整通过并推荐的图片

| COCO ID | 内容 | Split | 分辨率 | Flickr 署名 |
|---|---|---|---|---|
| 715 | 饮品与水果摊 | val2014 | 640 × 480 | zoetnet |
| 508 | 雪地山谷 | train2014 | 640 × 480 | dgrosso23 |
| 450 | 披萨 | train2014 | 640 × 480 | Roland Tanglao |
| 6730 | 猫与旅行箱 | train2014 | 640 × 480 | zenilorac |
| 292271 | 草地斑马 | train2014 | 640 × 480 | Alistair Young |
| 562382 | 湖边象群 | train2014 | 640 × 480 | shankar s. |

### COCO 715：饮品与水果摊

- COCO ID: 715
- File: `COCO_val2014_000000000715.jpg`
- Split: `val2014`
- Resolution: 640 × 480
- COCO license: 4 / Attribution License / [CC BY 2.0](http://creativecommons.org/licenses/by/2.0/)
- COCO URL: [下载原始 JPG](http://images.cocodataset.org/val2014/COCO_val2014_000000000715.jpg)
- Flickr source: [原图页面](https://www.flickr.com/photos/zoetnet/3732947167)；[annotation 中的静态图像 URL](http://farm4.staticflickr.com/3423/3732947167_879fffd0aa_z.jpg)
- Photographer/uploader: zoetnet（账号路径：zoetnet）
- Flickr title: smoothies
- Flickr license: [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/)
- Local file: [COCO_val2014_000000000715.jpg](images\COCO_val2014_000000000715.jpg)
- SHA-256: `45e8925104949b569b5280347cb829149dd3e9e9ed51098f3a55847ed7afa294`

### COCO 508：雪地山谷

- COCO ID: 508
- File: `COCO_train2014_000000000508.jpg`
- Split: `train2014`
- Resolution: 640 × 480
- COCO license: 4 / Attribution License / [CC BY 2.0](http://creativecommons.org/licenses/by/2.0/)
- COCO URL: [下载原始 JPG](http://images.cocodataset.org/train2014/COCO_train2014_000000000508.jpg)
- Flickr source: [原图页面](https://www.flickr.com/photos/damongrosso/6854522413)；[annotation 中的静态图像 URL](http://farm8.staticflickr.com/7047/6854522413_4c79538923_z.jpg)
- Photographer/uploader: dgrosso23（账号路径：damongrosso）
- Flickr title: Maine Valley
- Flickr license: [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/)
- Local file: [COCO_train2014_000000000508.jpg](images\COCO_train2014_000000000508.jpg)
- SHA-256: `31988bc646bfa6f95b40d6367c2d83efb75a7d3c1883019157efd3982105dab1`

### COCO 450：披萨

- COCO ID: 450
- File: `COCO_train2014_000000000450.jpg`
- Split: `train2014`
- Resolution: 640 × 480
- COCO license: 4 / Attribution License / [CC BY 2.0](http://creativecommons.org/licenses/by/2.0/)
- COCO URL: [下载原始 JPG](http://images.cocodataset.org/train2014/COCO_train2014_000000000450.jpg)
- Flickr source: [原图页面](https://www.flickr.com/photos/roland/5569738713)；[annotation 中的静态图像 URL](http://farm6.staticflickr.com/5098/5569738713_e926850eb0_z.jpg)
- Photographer/uploader: Roland Tanglao（账号路径：roland）
- Flickr title: Best Italian-style pizza in Vancouver is the Margherita at Nicli Antica Pizzeria - 032820114606
- Flickr license: [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/)
- Local file: [COCO_train2014_000000000450.jpg](images\COCO_train2014_000000000450.jpg)
- SHA-256: `0d8eb23c53e83ea805d280c8064ac4568ff56c24adfa1c3e0af3dca3f9f943aa`

### COCO 6730：猫与旅行箱

- COCO ID: 6730
- File: `COCO_train2014_000000006730.jpg`
- Split: `train2014`
- Resolution: 640 × 480
- COCO license: 4 / Attribution License / [CC BY 2.0](http://creativecommons.org/licenses/by/2.0/)
- COCO URL: [下载原始 JPG](http://images.cocodataset.org/train2014/COCO_train2014_000000006730.jpg)
- Flickr source: [原图页面](https://www.flickr.com/photos/zenilorac/195448192)；[annotation 中的静态图像 URL](http://farm1.staticflickr.com/75/195448192_af8971813f_z.jpg)
- Photographer/uploader: zenilorac（账号路径：zenilorac）
- Flickr title: timmy goes on holiday
- Flickr license: [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/)
- Local file: [COCO_train2014_000000006730.jpg](images\COCO_train2014_000000006730.jpg)
- SHA-256: `84541c8a41d46cbe35f730b687f68c814d31e43158d72d321fd669ea07b37ba3`

### COCO 292271：草地斑马

- COCO ID: 292271
- File: `COCO_train2014_000000292271.jpg`
- Split: `train2014`
- Resolution: 640 × 480
- COCO license: 4 / Attribution License / [CC BY 2.0](http://creativecommons.org/licenses/by/2.0/)
- COCO URL: [下载原始 JPG](http://images.cocodataset.org/train2014/COCO_train2014_000000292271.jpg)
- Flickr source: [原图页面](https://www.flickr.com/photos/ajy/3889833692)；[annotation 中的静态图像 URL](http://farm4.staticflickr.com/3536/3889833692_4da6615955_z.jpg)
- Photographer/uploader: Alistair Young（账号路径：ajy）
- Flickr title: Zebra at Leipzig Zoo
- Flickr license: [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/)
- Local file: [COCO_train2014_000000292271.jpg](images\COCO_train2014_000000292271.jpg)
- SHA-256: `404234598c037d6b93b28a88cf0eace39073dd96bd23bae2b4710fee72655b4f`

### COCO 562382：湖边象群

- COCO ID: 562382
- File: `COCO_train2014_000000562382.jpg`
- Split: `train2014`
- Resolution: 640 × 480
- COCO license: 4 / Attribution License / [CC BY 2.0](http://creativecommons.org/licenses/by/2.0/)
- COCO URL: [下载原始 JPG](http://images.cocodataset.org/train2014/COCO_train2014_000000562382.jpg)
- Flickr source: [原图页面](https://www.flickr.com/photos/shankaronline/7568363916)；[annotation 中的静态图像 URL](http://farm9.staticflickr.com/8428/7568363916_493c72ef0a_z.jpg)
- Photographer/uploader: shankar s.（账号路径：shankaronline）
- Flickr title: Elephants grazing
- Flickr license: [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/)
- Local file: [COCO_train2014_000000562382.jpg](images\COCO_train2014_000000562382.jpg)
- SHA-256: `38645946c6a7541bf6da30136751d41e324026075b6f9bb6d6322f2cc4dfe963`

## B. 原始清单中未完整通过的图片

| COCO ID | COCO CC BY 2.0 | SALICON 官方包 | Flickr 当前页 | 原因 |
|---|---|---|---|---|
| 89 | YES | YES | unavailable | Target photo absent from returned page; redirected to https://www.flickr.com/photos/// |
| 471 | YES | NO | unavailable | Not found in the complete 20,000-image official SALICON archive inventory; not a verified SALICON example; Target photo absent from returned page; redirected to https://www.flickr.com/photos/// |
| 573653 | YES | NO | verified | Not found in the complete 20,000-image official SALICON archive inventory; not a verified SALICON example |

- 89：COCO 许可通过，且属于 SALICON；Flickr 返回 Explore 页面，摄影者和当前许可留空，不猜测。
- 471：COCO 许可通过；未在本次官方 SALICON 20,000 张图片列表中找到，Flickr 原图页面也不可用。
- 471 的补充检索找到 [Wikimedia Commons 存档](https://commons.wikimedia.org/wiki/File:Crown_LAUSD_at_the_beach.jpg)，其中保留了旧 Flickr 地址和历史署名信息；按旧用户名及数字用户 ID 直接访问 Flickr 均为 HTTP 404。这是历史旁证，不代替当前 Flickr 核验，也不改变 SALICON 不通过的结果。
- 573653：COCO 许可和 Flickr 当前许可均为 CC BY 2.0（上传者页面名：In Memoriam: Andy / Andrew Fogg）；但未在上述 SALICON 图片列表中找到，不能作为已验证 SALICON 示例。
- 以上三张仍保存下载文件和 COCO 许可结论，但没有放入最终推荐清单。
- 补充筛选的其他尝试见 `metadata/exploratory_audit.json`。其中一些网页不可用或需要登录；另有低清晰度/主体遮挡的图片未选入最终展示。

## C. 论文使用与署名

- [SALICON 官方说明](https://www.salicon.net/challenge)明确其图像来自 COCO 2014。SALICON 显著性标注的 CC BY 4.0 与照片本身的许可不同，应分别处理。
- [COCO 使用条款](https://cocodataset.org/dataset/termsofuse.htm)说明 COCO 不拥有图片版权，不能仅引用数据集论文来替代照片署名。
- [CC BY 2.0 官方说明](https://creativecommons.org/licenses/by/2.0/deed.en)要求适当署名、附许可链接；有标题时保留标题，形成改编时说明改动。
- 将显著性热图叠加到原照片时，建议图注写为 `Adapted from [title], by [creator], CC BY 2.0; saliency overlay added.`，并链接原照片页面与许可证。若同时使用 SALICON 标注，还应保留其来源及对应许可。
- 投稿 Nature 前建议保留许可证页面截图或 HTML/PDF、照片来源页和本次 metadata 证据；本文件夹已保存 HTML 与哈希。未对 Nature 当期投稿细则作合规认定。
- 本次推荐的补充图已人工目视检查，无明显可识别人物。类别标注中无 person 只用作初筛，并不等同于人脸识别或法律判断。

### 可直接改写的英文署名

- COCO 715: “smoothies”, by zoetnet, [Flickr](https://www.flickr.com/photos/zoetnet/3732947167), [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/).
- COCO 508: “Maine Valley”, by dgrosso23, [Flickr](https://www.flickr.com/photos/damongrosso/6854522413), [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/).
- COCO 450: “Best Italian-style pizza in Vancouver is the Margherita at Nicli Antica Pizzeria - 032820114606”, by Roland Tanglao, [Flickr](https://www.flickr.com/photos/roland/5569738713), [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/).
- COCO 6730: “timmy goes on holiday”, by zenilorac, [Flickr](https://www.flickr.com/photos/zenilorac/195448192), [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/).
- COCO 292271: “Zebra at Leipzig Zoo”, by Alistair Young, [Flickr](https://www.flickr.com/photos/ajy/3889833692), [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/).
- COCO 562382: “Elephants grazing”, by shankar s., [Flickr](https://www.flickr.com/photos/shankaronline/7568363916), [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/).

## 文件与复查

- `images/`：原始候选与补充候选的 COCO JPG，本次记录共 9 张；最终推荐的 6 张见本页 A 节。
- `metadata/ccby2_manifest.csv`、`.json`：全部正式候选的结构化核验记录；`is_cc_by_2_0` 只表示 COCO 元数据许可，`recommended` 才表示本次最终推荐。
- `metadata/license_mapping.json`：实际读取的两个 split 的完整许可证映射。
- `metadata/official_annotation_excerpt.json`：直接从 instances JSON 取出的原始记录和许可证列表。
- `metadata/salicon_membership_source.json`：SALICON 成员身份的证据来源、HTTP 范围响应与文件数。
- `preview_contact_sheet.png`：最终 6 张的预览；仅等比例缩放和添加图外标签，不裁剪、不替代原图。
- `preview_requested_six.png`：用户原始 6 张的预览，可与 manifest 对照。
- `verify_and_download_ccby2.py`：可重新下载/解析官方标注并核验；网络请求最多重试 3 次，保留失败原因。

```bash
python verify_and_download_ccby2.py
python verify_and_download_ccby2.py --refresh
```

脚本核验使用 Python 标准库；预览和实际分辨率检查需要 Pillow。`--refresh` 更新网页证据，原始 JPG 如远端字节变化会报告错误而不覆盖。

轻量 ZIP 结果包包含 JPG、网页快照、元数据摘录、SALICON 中央目录证据、预览与脚本；为控制体积，不包含 253 MB 的完整 COCO 标注压缩包和全量衍生 metadata。本机完整文件夹仍保留它们。解压轻量包后首次运行脚本，会从官方 URL 自动补齐压缩包。

## 本次终端统计

```json
{
  "requested": 6,
  "requested_found_in_coco": 6,
  "requested_verified_license_id_4": 6,
  "requested_downloaded": 6,
  "requested_salicon_members": 4,
  "requested_flickr_pages_verified": 4,
  "requested_complete_verification": 3,
  "supplements_checked": 3,
  "final_recommended": 6,
  "conflicts": 0,
  "download_failures": 0,
  "original_incomplete": 3
}
```
